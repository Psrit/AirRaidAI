import os
import random
import time
import typing

import gymnasium as gym
import numpy as np
import tensorflow as tf

from config import (BATCH_SIZE, INITIAL_EPSILON, MAX_NUM_TIME_STEPS,
                    REPLAY_CAPACITY, REWARD_GAMMA, TIME_STEPS_PER_SAVE,
                    TIME_STEPS_PER_TRAIN, TRAIN_STEPS_PER_Q_SYNC, WARM_START,
                    epsilon)
from replaymemory import ObsvTransition, ReplayMemory
from utils import huber_loss, sync

Shape2DLike = typing.Union[int, typing.Tuple[int, int]]

LayerType = typing.TypeVar("LayerType", bound=tf.keras.layers.Layer)


class LayerConfig(object):
    def __init__(
        self,
        layer_type: typing.Type[LayerType],
        config: typing.Mapping[str, typing.Any]
    ):
        self.layer_type = layer_type
        self.config = config


def create_q_model(
    state_shape: typing.Tuple[int, ...],
    layer_configs: typing.Iterable[LayerConfig],
    state_dtype=tf.float32,
    is_target_network=False,
    name=None,
    **kwargs
) -> tf.keras.Model:
    """
    Create the `tf.keras.Model` of a Q network.

    :param state_shape:
        The shape of each input state.
    :param layer_configs:
        Configs of model layers, which are listed in order.
        Note that the last layer must be a 1D layer, whose length defines 
        the dimension of action space.
    :param state_dtype:
        Data type of input states.
    :param is_target_network:
        Tells whether this Q network is the trainable model or the target
        model of the DQN algorithm.
    :param name:
        Name of this network, used to create scope for network parameters.
    :param kwargs:
        Other keyword arguments that will be passed into the initializer of
        the `tf.keras.Model` class.

    :return:
        The created model (which is not compiled).

    """

    kwargs["trainable"] = (False if is_target_network else True)
    if name is None:
        name = "QNetwork" + ("-target" if is_target_network else "")
    kwargs["name"] = name

    inputs = tf.keras.layers.Input(shape=state_shape, dtype=state_dtype)
    x = inputs
    for layer_config in layer_configs:
        _layer = layer_config.layer_type(**layer_config.config)
        x = _layer(x)
    outputs = x
    model = tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)
    return model


class DQN(object):
    """
    A Deep Q Network class.

    """

    def __init__(
        self,
        state_shape: typing.Tuple[int, ...],
        layer_configs: typing.List[LayerConfig],
        reward_gamma: float = 0.9,
        epsilon: typing.Union[
            float, typing.Callable[[int], float]
        ] = INITIAL_EPSILON,
        optimizer: tf.optimizers.legacy.Optimizer = tf.keras.optimizers.legacy.RMSprop(),
        state_dtype=np.float32,
        train_steps_per_q_sync: int = 1
    ):
        """
        Initializes the DQN algorithm.

        :param state_shape:
            The shape of input state (which is generated from a ReplayMemory 
            instance).
        :param layer_configs:
            Configurations of layers, including the output layer.
            Note that the output dimension of the last layer defines the dimension
            of action space, i.e. number of actions in the game.
        :param reward_gamma:
            The reward discount, a float in [0, 1].
        :param epsilon:
            Generator of parameters of ε-greedy strategy, which returns an epsilon
            float (in [0, 1)) for a given integer (specifying the train step).
            If being a single float, then a constant generator lambda t: epsilon
            will be used as the DQN instance's epsilon function.
        :param optimizer:
            The optimizer for Q model training.
            It will be overridden if the models are loaded.
            [Important note] In tensorflow 2.11, `tf.keras.optimizers.Optimizer`
            points to a new base class, which leads to errors when it is loaded
            from disk (Ref: https://github.com/keras-team/keras-io/issues/1241).
            Hence `tf.optimizers.legacy.Optimizer`s are required here.
        :param state_dtype:
            Data type of input state.
        :param train_steps_per_q_sync:
            Number of training steps per synchronization from Q network to target
            Q network.

        """
        # configs for constructing the network
        self.state_shape = state_shape

        # optimizer
        self.optimizer = optimizer

        # initialize the models
        self.q_model: typing.Optional[tf.keras.models.Model] = None
        self.q_model_target: typing.Optional[tf.keras.models.Model] = None
        self.action_dim = None

        self.state_dtype = state_dtype
        self.initialize_model(layer_configs)

        # hyperparameters for DQN
        self.reward_gamma = reward_gamma
        self.epsilon = (epsilon if isinstance(epsilon, typing.Callable)
                        else (lambda t: epsilon))
        self.train_steps_per_q_sync = train_steps_per_q_sync

        # number of timesteps trained
        self.num_trained_steps = 0

        self.loss_history: typing.List[typing.Tuple[int, float]] = []

        self.checkpoint = tf.train.Checkpoint(
            q_model=self.q_model,
            q_model_target=self.q_model_target,
            optimizer=self.optimizer,
        )

    def _check_action_dim(self):
        output_shape = self.q_model.output_shape[1:]
        if len(output_shape) != 1:
            raise ValueError(
                f"The output layer is not 1-D, whose shape is {output_shape} "
                "(batch dimension not included)"
            )

        if self.action_dim is None:
            self.action_dim = output_shape[0]
        elif self.action_dim != output_shape[0]:
            raise ValueError(
                f"The output shape of q_model ({output_shape}) does not match "
                f"`action_dim` ({self.action_dim})"
            )

    def initialize_model(self, layer_configs):
        """
        Initializes the Q networks.

        """
        self.q_model = create_q_model(
            state_shape=self.state_shape,
            state_dtype=self.state_dtype,
            layer_configs=layer_configs,
            is_target_network=False,
            name="q-model"
        )
        self.q_model_target = create_q_model(
            state_shape=self.state_shape,
            state_dtype=self.state_dtype,
            layer_configs=layer_configs,
            is_target_network=True,
            name="q-model-target"
        )

        self._check_action_dim()

    def load(
        self,
        save_path: str
    ):
        """
        Load a saved DQN instance.

        In order to load a saved model, initialize the `DQN` class as usual
        (since some properties like model structure need to be defined through 
        `__init__`), and then call this method, i.e.
        ```
        dqn = DQN(...)
        dqn.load_model(save_path, custom_objects={})
        ```

        After this the weights of Q networks (`q_model` and `q_model_target`)
        are restored, along with the optimizer and hyperparameters of the DQN 
        instance.
        Note that only a single `epsilon` float can be restored, hence 
        `self.epsilon` will be a constant epsilon function, which can be further
        modified manually after the loading completed.

        :param save_path:
            The path to the directory where the state of the DQN instance and 
            the history of loss values are saved.

        """
        # load Q models and optimizer
        manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, directory=save_path, max_to_keep=3
        )
        self.checkpoint.restore(manager.latest_checkpoint)

        # load hyperparameters
        hyperparams_path = os.path.join(save_path, "hyperparams.npz")
        npz = np.load(hyperparams_path)

        self.reward_gamma = npz["reward_gamma"]
        self.epsilon = lambda t: npz["epsilon"]
        self.train_steps_per_q_sync = npz["train_steps_per_q_sync"]

        # check and set self.action_dim
        # self._check_action_dim()  # already performed in __init__

    def save(self, save_path='training_savings'):
        """
        Save the weights of Q networks (`q_model` and `q_model_target`), the 
        state of the optimizer, hyperparameters of the DQN instance and record
        of loss values.
        Note that `self.epsilon` will be stored as a single value, i.e.
        `self.epsilon(self.train_steps)`. 

        :param save_path: str
            The path to the directory where the state of the DQN instance and 
            the history of loss values are saved.
            If not existed, the directory will be created.

        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # save weights of Q networks and the optimizer
        manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, directory=save_path, max_to_keep=3
        )
        manager.save()

        # save hyperparameters
        hyperparams_path = os.path.join(save_path, "hyperparams.npz")
        np.savez_compressed(
            hyperparams_path,
            reward_gamma=self.reward_gamma,
            epsilon=self.epsilon(self.num_trained_steps),
            train_steps_per_q_sync=self.train_steps_per_q_sync
        )

        # save the loss values
        loss_records_path = os.path.join(save_path, "loss_records.npz")
        if os.path.exists(loss_records_path):
            loss_npzfile = np.load(loss_records_path)
            loss_records = dict(loss_npzfile)
        else:
            loss_records = {}
        loss_records[
            f"loss_{time.strftime('%Y%m%d_%H%M%S')}"
        ] = np.array(self.loss_history)

        np.savez_compressed(
            loss_records_path,
            **loss_records,
        )

        # reset the loss history since the values have been stored
        self.loss_history: typing.List[typing.Tuple[int, float]] = []

    @tf.function
    def _td_error(
        self,
        b_state,
        b_action,
        b_reward,
        b_done,
        b_state_
    ) -> np.ndarray:
        """
        Calculate time difference error according to
            L = Q_{train}(s_j, a_j) - Q_j
        where
            Q_j = r_j                                       for terminal s_{j+1};
                  r_j + γ max_a { Q_{target}(s_{j+1}, a) }  for non-terminal s_{j+1},

        where "max_a {f}" means calculating the maximal value of f over all a
        values.

        :param b_state:
            Batch of states (shape=(batch_size, *state_shape)).
        :param b_action:
            Batch of actions (shape=(batch_size,)).
        :param b_reward:
            Batch of rewards (shape=(batch_size,)).
        :param b_done:
            Batch of booleans which tell whether s_{j+1} is a terminal step
            (i.e. done) or not (shape=(batch_size,)).
        :param b_state_:
            Batch of next states (shape=(batch_size, *state_shape)).

        """
        b_q_ = (1 - b_done) * \
            tf.reduce_max(self.q_model_target(b_state_), axis=1)
        b_q = tf.reduce_sum(self.q_model(b_state) *
                            tf.one_hot(b_action, depth=self.action_dim), axis=1)
        # shape=(batch_size,)
        return b_q - (b_reward + self.reward_gamma * b_q_)

    def train(self, b_state, b_action, b_reward, b_done, b_state_):
        """
        Train the Q-network with a batch of transitions:

            state --action--> reward, done, state_

        where state_ indicates the next state.

        """
        loss = self._train(b_state, b_action, b_reward, b_done, b_state_)

        self.loss_history.append((self.num_trained_steps, loss))

        self.num_trained_steps += 1

        if self.num_trained_steps % self.train_steps_per_q_sync == 0:
            sync(self.q_model, self.q_model_target)
            print("Target Q network parameters synchronized.")

    @tf.function
    def _train(self, b_state, b_action, b_reward, b_done, b_state_):
        with tf.GradientTape() as tape:
            td_errors = self._td_error(
                b_state, b_action, b_reward, b_done, b_state_
            )
            loss = tf.reduce_mean(huber_loss(td_errors))

        grad = tape.gradient(loss, self.q_model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grad, self.q_model.trainable_weights)
        )
        return loss

    def select_action(self, state):
        """
        Given current state, selects the next action according to ε-greedy
        strategy, i.e. with probablity ε selects a random action from the
        action space, otherwise selects the action as argmax_a{Q(state, a)},
        where Q is the **(trainable) Q network**.

        :param state:
            Current state (not a batch of states, i.e. no batch-dimension).

        :return: 
            Next action.

        """
        if random.random() <= self.epsilon(self.num_trained_steps):
            action = random.randint(0, self.action_dim - 1)
        else:
            # Here batch_size = 1, so self.q_model(state).shape =
            # [1, action_dim].
            state = np.expand_dims(state, 0).astype(self.state_dtype)
            q_value = self.q_model(state).numpy()[0]  # shape=(action_dim,)
            action = np.argmax(q_value)

        return action


def run_dqn_algorithm(
    env: gym.Env,
    layer_configs: typing.List[LayerConfig],
    reward_gamma: float = REWARD_GAMMA,
    epsilon: typing.Union[float, typing.Callable[[int], float]] = epsilon,
    preprocess: typing.Optional[typing.Callable] = None,
    preprocessed_obsv_shape: typing.Optional[typing.Tuple[int, ...]] = None,
    replay_capacity: int = REPLAY_CAPACITY,
    hist_len: int = 1,
    hist_type: str = "linear",
    hist_spacing: int = 1,
    max_sample_attempts: int = 1000,
    batch_size: int = BATCH_SIZE,
    load_path: typing.Optional[str] = None,
    save_path: str = "saved_dqn",
    num_time_steps: int = MAX_NUM_TIME_STEPS,
    time_steps_per_train: int = TIME_STEPS_PER_TRAIN,
    train_steps_per_q_sync: int = TRAIN_STEPS_PER_Q_SYNC,
    time_steps_per_save: int = TIME_STEPS_PER_SAVE,
    warm_start: int = WARM_START
):
    """
    Run DQN algorithm on a given environment.

    :param env:
        The game environment.
    :param preprocess:
        Preprocessor of env observations.
    :param observation_shape:
        Shape of preprocessed observations.
    :param layer_configs:
        Configurations of layers, including the output layer.
        Note that the output dimension of the last layer defines the dimension
        of action space, i.e. number of actions in the game, which must be 
        consistent with `env.action_space.n`.
    :param reward_gamma:
        The reward discount, a float in [0, 1].
    :param epsilon:
        Generator of parameters of ε-greedy strategy, which returns an epsilon
        float (in [0, 1)) for a given integer (specifying the train step).
        If being a single float, then a constant generator lambda t: epsilon
        will be used as the DQN instance's epsilon function.
    :param preprocess:
        Preprocessor for raw observation from `env`, output of which will be 
        stored into the `ReplayMemory` object.
        Being None means no preprocess will be performed.
    :param preprocessed_obv_shape:
        The shape of the output of `preprocess`, which determines
        `ReplayMemory.observation_shape`.
        When `preprocess` is None, this value will be the shape of raw 
        observations from `env`. 
    :param replay_capacity:
        The capacity (i.e. max size) of the replay memory.
    :param hist_len:
    :param hist_type:
    :param hist_spacing:
    :param max_sample_attempts:
        Parameters for constructing replay memory. Please refer to 
        the docstring of `ReplayMemory`.
    :param batch_size:
        The size of batches used in `self.train`.
    :param load_path:
        Path to the directory where the state of DQN instance is to be loaded.
        If given, the instance will be loaded before training; if leaving to be
        None, the instance created from scratch will be used directly.
    :param save_path:
        Path to the directory where the state of DQN instance is to be saved.
    :param num_time_steps:
        Total number of time steps.
    :param time_steps_per_train:
        Number of time steps per training of Q network.
    :param train_steps_per_q_sync:
        Number of training steps per synchronization from Q network to target 
        Q network.
    :param time_steps_per_save:
        Number of time steps per saving of the state of DQN instance.
    :param warm_start:
        When time_step < warm_start, no training nor saving will happen.

    """
    env.reset()

    if preprocess is None:
        def preprocess(_): return _
        observation_shape = env.observation_space.shape
    else:
        if preprocessed_obsv_shape is None:
            raise ValueError(
                "`preprocessed_obv_shape` can not be None "
                "when `preprocess` function is specified."
            )
        observation_shape = preprocessed_obsv_shape

    replay_memory = ReplayMemory(
        observation_shape=observation_shape,
        capacity=replay_capacity,
        hist_len=hist_len,
        hist_type=hist_type,
        hist_spacing=hist_spacing,
        max_sample_attempts=max_sample_attempts,
        dtype=env.observation_space.dtype
    )
    dqn = DQN(
        state_shape=replay_memory.state_shape,
        layer_configs=layer_configs,
        reward_gamma=reward_gamma,
        epsilon=epsilon,
        train_steps_per_q_sync=train_steps_per_q_sync
    )
    if load_path:
        dqn.load(load_path)
        print(f"Load DQN state from {os.path.abspath(load_path)}")

    o = env.reset()[0]
    o = preprocess(o)
    episode = 0
    ep_length = 0
    ep_return = 0

    for time_step in range(1, num_time_steps + 1):
        a = dqn.select_action(replay_memory.construct_current_state(o))
        o_, r, terminated, truncated, info = env.step(a)
        o_ = preprocess(o_)
        replay_memory.add(ObsvTransition(o, a, r, terminated, o_))

        if time_step >= warm_start and time_step % time_steps_per_train == 0:
            b_state, b_action, b_reward, b_done, b_state_ = \
                replay_memory.sample(batch_size)
            dqn.train(b_state, b_action, b_reward, b_done, b_state_)

        if time_step >= warm_start and time_step % time_steps_per_save == 0:
            dqn.save(save_path)
            print(f"Save DQN state to {os.path.abspath(save_path)}")

        ep_length += 1
        ep_return += r

        if terminated:
            o = env.reset()[0]
            o = preprocess(o)

            # TODO: check the criterion for the end of episode
            episode += 1
            print(
                'Time steps so far: {}, episode so far: {}, '
                'episode return: {:.4f}, episode length: {}'
                .format(time_step, episode, ep_return, ep_length)
            )

            ep_length = 0
            ep_return = 0
        else:
            o = o_
