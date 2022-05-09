import getopt
import sys
import typing

import gym
import numpy as np
import tensorflow as tf
# from gym.wrappers import Monitor
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from defaults import (BATCH_SIZE, NUM_EPISODES, NUM_EPISODES_FOR_NEXT_SAVE,
                      NUM_STEPS_PER_EPISODE, REPLAY_CAPACITY)
from dqn import DQN, LayerConfig
from gym_atari import EnvProtocol

ImageLike = typing.Union[np.ndarray, Image.Image, tf.Tensor]


# For every `NUM_EPISODES_FOR_NEXT_TEST_ROUND` episodes, conduct one round of
# test, which includes `NUM_TESTS_PER_TEST_ROUND` tests.
# In each test, we play the game for `NUM_STEPS_PER_EPISODE` steps (if the game
# is not lost).
NUM_EPISODES_FOR_NEXT_TEST_ROUND = 5
NUM_TESTS_PER_TEST_ROUND = 5


class AirRaidPreprocessor:
    # shape in format (width, height, n_channels)
    env_observation_shape = (160, 250, 3)
    pframe_shape = (80, 80, 1)  # (width, height)

    def __call__(self, frame: ImageLike) -> np.ndarray:
        if isinstance(frame, Image.Image):
            im = frame
        elif isinstance(frame, (np.ndarray, tf.Tensor)):
            frame = tf.convert_to_tensor(frame)
            frame = np.asarray(frame)
            im = Image.fromarray(frame)
        else:
            raise ValueError(
                f"Unsupported state type: `{type(frame)}`. "
                f"Only `PIL.Image.Image`, `np.ndarray` and `tf.Tensor` "
                f"are supported."
            )

        im = im.convert("L")
        im.thumbnail((80, 125), Image.ANTIALIAS)  # shape=(80, 125)

        width, height = im.size
        # Crop the image according to the (l, r, u, d) borders:
        im = im.crop((0, height - 80, width, height))  # shape=(80, 80)

        pframe = np.asarray(im) / 255.0  # shape=(80, 80)
        pframe = np.expand_dims(pframe, -1)  # shape=(80, 80, 1)

        return pframe


def create_airraid_dqn(airraid_env, load_save_path=None):
    airraid_preprocessor = AirRaidPreprocessor()

    # initialize DQN
    layer_configs = (
        LayerConfig(Conv2D, dict(
            kernel_size=(8, 8), filters=16, strides=4,
            activation="relu",
            name="conv1"
        )),
        LayerConfig(Conv2D, dict(
            kernel_size=(4, 4), filters=32, strides=1,
            activation="relu",
            name="conv2"
        )),
        LayerConfig(Flatten, {}),
        LayerConfig(Dense, dict(units=256, activation="relu", name="fc1")),
        LayerConfig(Dense, dict(units=airraid_env.action_space.n,
                                activation=None, name="q_layer"))
    )
    network = DQN(
        pframe_shape=airraid_preprocessor.pframe_shape,
        layer_configs=layer_configs,
        frame_preprocessor=airraid_preprocessor,
        replay_capacity=max(4 + BATCH_SIZE * 3, REPLAY_CAPACITY),
        hist_len=15,
        hist_type="linear",
        hist_spacing=2,
        batch_size=BATCH_SIZE,
        time_steps_per_train=64
    )
    if load_save_path is not None:
        try:
            network.load_model(save_path=load_save_path)
        except OSError as e:
            print(e)
            print("Create a new model")
        except Exception as e:
            print(type(e), e)
        else:
            print(f"Model loaded from {load_save_path}")

    network.q_model.summary()
    return network


def train_airraid(train_display=False, test_display=True,
                  save_path="airraid-training-saves"):
    if train_display:
        env = gym.make('ALE/AirRaid-v5', render_mode="human")
    else:
        env: EnvProtocol = gym.make('ALE/AirRaid-v5')

    if test_display:
        test_env = gym.make('ALE/AirRaid-v5', render_mode="human")
    else:
        test_env: EnvProtocol = gym.make('ALE/AirRaid-v5')

    losses: typing.List[typing.Tuple[int, float]] = []
    rewards: typing.List[typing.Tuple[int, float]] = []

    network: DQN = create_airraid_dqn(env, save_path)

    for episode in range(1, NUM_EPISODES + 1):
        print("========== Episode {0} ==========".format(episode))
        observation = env.reset()
        done = False
        # Go `NUM_STEPS_PER_EPISODE` steps in every episode
        for step in range(NUM_STEPS_PER_EPISODE):
            # now `done` must be False
            action = network.select_action_from_frame(observation)
            # print(action)
            new_observation, reward, done, _ = env.step(action)
            network.perceive(
                observation, action, reward, done, new_observation
            )
            observation = new_observation
            # If the game ends, stop training steps of this episode immediately.
            if done:
                break

        evaluated_loss_value = network.evaluate()
        print('[TRAIN] After Episode {0: 3d}, evaluated loss value = {1:.3f}'
              .format(episode, evaluated_loss_value))

        # Save the model after `NUM_EPISODES_FOR_NEXT_SAVE` episodes
        if episode % NUM_EPISODES_FOR_NEXT_SAVE == 0:
            losses += network.loss_history
            network.save(save_path)

        # Test `NUM_TESTS_PER_TEST_ROUND` times for every `NUM_EPISODES_FOR_NEXT_TEST_ROUND` episodes
        if episode % NUM_EPISODES_FOR_NEXT_TEST_ROUND == 0:
            total_reward = 0.0
            for i in range(NUM_TESTS_PER_TEST_ROUND):
                print("[TEST] Test: " + str(i))
                _reward_in_round = 0
                observation = test_env.reset()
                done = False
                for j in range(NUM_STEPS_PER_EPISODE):
                    action = network.select_action_from_frame(observation)
                    observation, reward, done, _ = test_env.step(action)
                    _reward_in_round += reward
                    if done:
                        break
                print(f"reward = {_reward_in_round}")
                total_reward += _reward_in_round
            aver_reward = total_reward / NUM_TESTS_PER_TEST_ROUND
            rewards.append((network.time_step, aver_reward))
            print('[TEST] After Episode {0: 3d}, evaluated average reward = {1:.3f}'
                  .format(episode, aver_reward))

    env.close()
    test_env.close()

    return losses, rewards


def play_airraid(play_rounds=100, display=False, load_save_path=None):
    if display:
        env = gym.make('ALE/AirRaid-v5', render_mode="human")
    else:
        env = gym.make('ALE/AirRaid-v5')
    network: DQN = create_airraid_dqn(env, load_save_path)

    total_reward = 0.0
    for i in range(play_rounds):
        observation = env.reset()
        done = False
        while not done:
            action = network.select_action_from_frame(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
    aver_reward = total_reward / play_rounds
    print('Average reward in {0} rounds of game: {1}'
          .format(play_rounds, aver_reward))

    env.close()


if __name__ == '__main__':
    # TODO: improve the parser of the command line parameters.
    mode = ''
    try:
        ovpairs, args = getopt.getopt(sys.argv[1:], "hm:", ["help", "mode"])
    except getopt.GetoptError:
        print("Invalid argument!")
        sys.exit(1)
    else:
        if len(ovpairs) != 0:
            for opt, val in ovpairs:
                if opt in ("-h", "--help"):
                    print("Usage: python airraid.py [-h|-m t|-m p]")
                    sys.exit()
                elif opt in ("-m", "--mode"):
                    mode = val
        else:
            mode = 't'

    if mode in ("t", "train"):
        losses, rewards = train_airraid(train_display=False, test_display=True)
        print(f"Losses:\n{losses}")
        print(f"Rewards:\n{rewards}")
    elif mode in ("p", "play"):
        play_airraid(play_rounds=5, display=True,
                     load_save_path="airraid-training-saves")
    else:
        print("Invalid argument! Mode can only be 't' or 'p', but '{0}' "
              "received.".format(mode))
