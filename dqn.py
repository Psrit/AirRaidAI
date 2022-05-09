import os
import random
import time
import typing

import numpy as np
import tensorflow as tf

from defaults import (BATCH_SIZE, REPLAY_CAPACITY, TIME_STEPS_PER_TRAIN,
                      TRAIN_STEPS_PER_Q_SYNC)
from replaymemory import FrameTransition, ReplayMemory

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

    inputs = tf.keras.layers.Input(shape=state_shape)
    x = inputs
    for layer_config in layer_configs:
        _layer = layer_config.layer_type(**layer_config.config)
        x = _layer(x)
    outputs = x
    model = tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)
    return model


class DQN(object):
    """
    A deep Q network class.

    To use this network, please follow the steps:

    1. initialize DQN with algorithm hyperparameters and configurations of 
    convolution layers and fully connected layers (which define the structure
    of the network).

    2. use `perceive` to make the network add state transition into the replay
    memory and train parameters if needed.

    3. call `select_action_from_frame` to use the network to choose the next 
    action according to current frame.

    """

    INITIAL_EPSILON = 0.5
    FINAL_EPSILON = 0.01
    GAMMA = 0.9

    def __init__(self,
                 pframe_shape: typing.Tuple[int, ...],
                 layer_configs: typing.Union[
                     typing.List[LayerConfig], None
                 ] = None,
                 load_save_path: typing.Optional[str] = None,
                 custom_objects: typing.Optional[typing.Mapping[str, typing.Any]] = None,
                 frame_preprocessor: typing.Callable = lambda x: x,
                 loss: tf.keras.losses.Loss = tf.keras.losses.mse,
                 optimizer: tf.optimizers.Optimizer = tf.keras.optimizers.RMSprop(),
                 replay_capacity: int = REPLAY_CAPACITY,
                 hist_len: int = 1,
                 hist_type: str = "linear",
                 hist_spacing: int = 1,
                 max_sample_attempts: int = 1000,
                 batch_size: int = BATCH_SIZE,
                 time_steps_per_train: int = TIME_STEPS_PER_TRAIN,
                 train_steps_per_q_sync: int = TRAIN_STEPS_PER_Q_SYNC,
                 pframe_dtype=np.float32):
        """
        Initializes the DQN algorithm.

        :param pframe_shape:
            The shape of input preprocessed frame.
            For example, if each frame is a preprocessed 3-channel image, then 
            the shape is (width, height, 3).
        :param layer_configs:
            Configurations of layers.
            Note that the output dimension of the last layer defines the dimension
            of action space, i.e. number of actions in the game.
            If being None, the models will be loaded from `load_save_path`.
            See the explanation of `load_save_path`.
        :param load_save_path:
            If not None, the models will be loaded from `load_save_path`.
            Note that arguments `layer_configs` and `load_save_path` are related
            as:
                if load_save_path is not None:
                    layer_configs can be None and will be ignored;
                    q_model (and the optimizer and loss function) and
                    q_model_target are loaded from load_save_path. 
                if load_save_path is None:
                    layer_configs cannot be None;
                    q_model and q_model_target are created according to 
                    layer_configs.
        :param custom_objects:
            Optional dictionary mapping names (strings) to custom classes or 
            functions (like custom Layer classes or loss functions) to be 
            considered during deserialization.
            Only works when `load_save_path` is not None.
        :param frame_preprocessor:
            A function that can preprocess the frame yielded from `env`.
            For example, this callable may convert the image from RGB
            representation to gray-scale representation, and then down-sample
            it.
        :param loss:
            The loss function for model training.
            It will be overridden if the models are loaded from `load_save_path`.
        :param optimizer:
            The optimizer for model training.
            It will be overridden if the models are loaded from `load_save_path`.
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
        :param time_steps_per_train:
            Train once after perceiving if time_step % time_steps_per_train == 0.
        :param train_steps_per_q_sync:
            Synchronize the weights of the target Q model using the weights of 
            the trainable Q model after training if
                train_step % train_steps_per_q_sync == 0.
        :param pframe_dtype:
            The data type of preprocessed frames in the replay memory and
            hence the data type of input data of Q model in the algorithm.

        """
        # configs for constructing the network
        self.pframe_shape = pframe_shape

        # components for compiling the models
        self.loss = loss
        self.optimizer = optimizer

        # initialize replay memory
        # state --action--> reward, done, next_state
        self.pframe_dtype = pframe_dtype
        self.replay_memory = ReplayMemory(
            frame_shape=self.pframe_shape,
            capacity=replay_capacity,
            hist_len=hist_len,
            hist_type=hist_type,
            hist_spacing=hist_spacing,
            max_sample_attempts=max_sample_attempts,
            dtype=self.pframe_dtype
        )

        # initialize and compile the models
        self.q_model: typing.Optional[tf.keras.models.Model] = None
        self.q_model_target: typing.Optional[tf.keras.models.Model] = None
        self.action_dim = None
        if load_save_path is not None:
            self.load_model(load_save_path, custom_objects=custom_objects)
        elif layer_configs is not None:
            self.initialize_model(layer_configs)
        else:
            raise ValueError(
                "`layer_configs` and `load_save_path` cannot be both None."
            )

        # hyperparameters for DQN algorithm
        self.time_step = 0
        self.train_step = 0
        self.epsilon = self.INITIAL_EPSILON  # a float in [0, 1)

        # env-frame preprocessor
        self._frame_preprocessor = frame_preprocessor

        self.batch_size = batch_size
        self.time_steps_per_train = time_steps_per_train
        self.train_steps_per_q_sync = train_steps_per_q_sync

        # caches for current preprocessed frame
        # if it is not None, it must correspond to a non-terminal frame
        # (refer to self.perceive)
        self.current_pframe = None

        self.loss_history: typing.List[typing.Tuple[int, float]] = []

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
        Initializes the networks.

        """
        self.q_model = create_q_model(
            state_shape=(self.replay_memory.hist_len, *self.pframe_shape),
            layer_configs=layer_configs,
            is_target_network=False,
            name="q-model"
        )
        self.q_model_target = create_q_model(
            state_shape=(self.replay_memory.hist_len, *self.pframe_shape),
            layer_configs=layer_configs,
            is_target_network=True,
            name="q-model-target"
        )

        # # connect all layers in `q_model` so that its `output_shape` can be
        # # accessed
        # self.q_model(tf.keras.layers.Input(self.pframe_shape))

        self.q_model.compile(
            optimizer=self.optimizer,
            loss=self.loss
        )

        self._check_action_dim()

    def load_model(
        self,
        save_path: str,
        custom_objects: typing.Optional[typing.Mapping[str, typing.Any]] = None
    ):
        """
        Load a saved model.

        In order to load a saved model, initialize the `DQN` class as usual
        (since some properties need to be defined through `__init__`), and
        then call this method, i.e.
        ```
        network = DQN(...)
        network.load_model(save_path, custom_objects={})
        ```
        or designating the save_path directly in `__init__`:
        ```
        network = DQN(load_save_path=save_path, ...)
        ```
        After this the models of `network` (`q_model` and `q_model_target`)
        are restored.

        :param save_path:
            The path to the directory where the model and the history of 
            loss values are saved.
            Its contents should be:
                save_path/ ─┬─ model
                            └─ loss_values

        :param custom_objects:
             Optional dictionary mapping names (strings) to custom classes or 
             functions (like custom Layer classes or loss functions) to be 
             considered during deserialization.

        """
        if not save_path.endswith(os.path.sep):
            save_path += os.path.sep
        model_path = save_path + "model"

        self.q_model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=True
        )
        self.q_model.trainable = True

        self.q_model_target = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
        self.q_model_target.trainable = False

        self._check_action_dim()

    def save(self, save_path='training_savings'):
        """
        Save the model (including its architecture, weights and the state of 
        the optimizer).

        :param save_path: str
            The path to the directory where the model and the history of 
            loss values are saved.
            If not existed, the directory will be created, and its contents
            will be:
                save_path/ ─┬─ model
                            └─ loss_records.npz

        """
        if not save_path.endswith(os.sep):
            save_path += os.sep
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        model_path = save_path + "model"
        loss_records_path = save_path + "loss_records.npz"

        # save the Q model
        self.q_model.save(model_path, save_format="h5")

        # save the loss values
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

    def preprocess_frame(
        self,
        frame
    ) -> np.ndarray:
        """
        Preprocess a frame yielded from the environment.

        :param frame:
            The input frame yielded from the environment, e.g. a 3-channel image
            whose shape is (width, height, 3).

        :return:
            The preprocessed frame `pframe`.

        """
        pframe = self._frame_preprocessor(frame)
        return pframe

    def perceive(
        self,
        frame,
        action: int,
        reward: float,
        done: bool,
        next_frame
    ):
        """
        Makes the Q networks perceive the transition relation:

            frame --action--> reward, done, next_frame

        i.e. stores (pframe, action, reward, done, next_pframe) into
        `self.replay_memory` (where pframe = preprocessed frame) and
        trains `q_model` if there are enough records in `self.replay_memory`.

        :param frame:
            Current frame generated from the environment.
        :param action:
            Action that yields `next_frame` from `frame`.
        :param reward:
            Reward of the action.
        :param done:
            Tells whether `next_frame` is a terminal frame.
        :param next_frame:
            Next frame.

        """
        self.time_step += 1

        if self.current_pframe is not None:
            # Directly use cached pframe since the transition is continuous,
            # i.e. self.current_pframe == self.preprocess_frame(frame)
            pframe = self.current_pframe
        else:
            # No cache available (the game just (re)started)
            pframe = self.preprocess_frame(frame)

        if not done:
            # Cache next_pframe as current pframe
            self.current_pframe = self.preprocess_frame(next_frame)
        else:
            # The game terminates, and clear cache
            self.current_pframe = None

        self.replay_memory.add(
            FrameTransition(pframe, action, reward, done)
        )

        if (len(self.replay_memory.sampleable_range()) > self.batch_size
                and self.time_step % self.time_steps_per_train == 0):
            self.train()

    @staticmethod
    def calculate_y(
        reward_batch: np.ndarray,
        next_q_value_batch: np.ndarray,
        is_next_state_done_batch: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        Calculate y_j according to the formula:

            y_j = r_j                                 for terminal φ_{j+1};
                  r_j + γ max_a { Q(φ_{j+1}, a) }     for non-terminal φ_{j+1},

        where "max_a {f}" means calculating the maximal value of f over all a
        values.

        :param reward_batch:
            Batch of rewards (shape=(batch_size,)).
        :param next_q_value_batch:
            Q values of next step, which should be evaluated using the **target
            Q model**.
            Note that only for the case where φ_{j+1} is non-terminal there is
            a valid Q(φ_{j+1}, a) value, therefore the shape of this argument is
            (num_not_done_cases_in_batch, action_dim).
        :param is_next_state_done_batch:
            Batch of booleans which tell whether φ_{j+1} is a terminal step
            (i.e. done) or not (shape=(batch_size,)).
            Note that
                len(np.where(logical_not(is_next_state_done_batch))[0])
                == num_not_done_cases_in_batch.
        :param gamma:
            Discounting factor of future rewards.

        """
        y_batch = reward_batch
        y_batch[np.logical_not(is_next_state_done_batch)] += (
            # for next_state not done:
            gamma * np.max(next_q_value_batch, axis=-1)
        )  # shape=(num_not_done_cases_in_batch,)
        return y_batch  # shape=(batch_size,)

    def train(self):
        print(
            "[ Training round {0:03d} ] ready to train..."
            .format(self.train_step)
        )

        # Step 1: generate random minibatch from replay memory
        (
            state_batch,  # shape=(batch_size, state_shape)
            action_batch,  # shape=(batch_size,)
            reward_batch,  # shape=(batch_size,)
            done_batch,  # shape=(batch_size,)
            next_state_batch  # shape=(batch_size, state_shape)
        ) = self.replay_memory.sample(self.batch_size)

        # Step 2: calculate y
        next_q_value_batch: np.ndarray = self.q_model_target(
            next_state_batch[np.logical_not(done_batch)],
            training=False
        )  # shape=(num_not_done_cases, action_dim)
        y_batch = self.calculate_y(
            reward_batch,
            next_q_value_batch,
            done_batch,
            self.GAMMA
        )  # shape=(batch_size,)

        # Since the output of `q_model` has a dimension of (batch_size, action_dim),
        # which does not match the dimension of `y_batch`, we should extend
        # `y_batch` so that the loss function can evaluates:
        #   loss(
        #       y_batch, q_model(state_batch)[np.arange(0, batch_size), action_batch]
        #   )
        y_batch_extended = tf.tensor_scatter_nd_update(
            self.q_model(
                state_batch,
                training=False
            ),
            tf.where(
                tf.one_hot(action_batch, depth=self.action_dim)
            ),  # shape=(batch_size, action_dim)
            y_batch
        )

        loss_value = self.q_model.train_on_batch(
            x=state_batch,
            y=y_batch_extended
        )
        print(f"time step: {self.time_step: 3d}  loss: {loss_value:.4f}")
        self.loss_history.append((self.time_step, loss_value))

        self.train_step += 1

        if self.train_step % self.train_steps_per_q_sync == 0:
            for train_layer, target_layer in zip(
                self.q_model.layers, self.q_model_target.layers
            ):
                target_layer.set_weights(train_layer.get_weights())
            print("Target Q network parameters synchronized.")

        if self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON -
                             self.FINAL_EPSILON) / 10000.0

    def evaluate(self):
        # Step 1: generate random minibatch from replay memory
        (
            state_batch,  # shape=(batch_size, state_shape)
            action_batch,  # shape=(batch_size,)
            reward_batch,  # shape=(batch_size,)
            done_batch,  # shape=(batch_size,)
            next_state_batch  # shape=(batch_size, state_shape)
        ) = self.replay_memory.sample(self.batch_size)

        # Step 2: calculate y
        next_q_value_batch: np.ndarray = self.q_model_target(
            next_state_batch[np.logical_not(done_batch)],
            training=False
        )  # shape=(num_not_done_cases, action_dim)
        y_batch = self.calculate_y(
            reward_batch,
            next_q_value_batch,
            done_batch,
            self.GAMMA
        )  # shape=(batch_size,)

        # Since the output of `q_model` has a dimension of (batch_size, action_dim),
        # which does not match the dimension of `y_batch`, we should extend
        # `y_batch` so that the loss function can evaluates:
        #   loss(
        #       y_batch, q_model(state_batch)[np.arange(0, batch_size), action_batch]
        #   )
        y_batch_extended = tf.tensor_scatter_nd_update(
            self.q_model(
                state_batch,
                training=False
            ),
            tf.where(
                tf.one_hot(action_batch, depth=self.action_dim)
            ),  # shape=(batch_size, action_dim)
            y_batch
        )

        evaluated_loss_value = self.q_model.evaluate(
            x=state_batch,
            y=y_batch_extended,
            batch_size=self.batch_size,
            verbose=0
        )

        return evaluated_loss_value

    def select_action(self, state):
        """
        Given current state, selects the next action according to ε-greedy
        strategy, i.e. with probablity ε selects a random action from the
        action space, otherwise selects the action as argmax_a{Q(state, a)},
        where Q is the **(trainable) Q model**.

        :param state:
            Current state. Note that the state here is actually a single 
            observation of environment, not a batch of states.

        :return: 
            Next action.

        """
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            # Here batch_size = 1, so self.q_value_layer(state).shape =
            # [1, action_dim].
            q_value = self.q_model(
                np.expand_dims(state, 0)
            ).numpy()[0]  # shape=(action_dim,)
            action = np.argmax(q_value)

        return action

    def select_action_from_frame(self, current_frame):
        """
        Given current frame, selects the next action according to ε-greedy
        strategy.

        :param current_frame:
            Current frame, which is not preprocessed and not added in the 
            replay memory yet.

        :return:
            Next action.

        """
        current_state = self.replay_memory.construct_current_state(
            self.preprocess_frame(current_frame)
        )
        return self.select_action(current_state)
