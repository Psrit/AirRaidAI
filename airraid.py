import getopt
import sys
import typing

import numpy as np
import tensorflow as tf
# from gym.wrappers import Monitor
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from config import (BATCH_SIZE, REPLAY_CAPACITY, TIME_STEPS_PER_SAVE,
                    TIME_STEPS_PER_TRAIN, TRAIN_STEPS_PER_Q_SYNC)
from dqn import DQN, LayerConfig, run_dqn_algorithm
from gym_interface import EnvProtocol, gym

ImageLike = typing.Union[np.ndarray, Image.Image, tf.Tensor]


class AirRaidPreprocessor:
    env_obsv_shape = (160, 250, 3)  # shape=(width, height, n_channels)
    preprocessed_obsv_shape = (80, 80, 1)  # shape=(width, height)

    def __call__(self, obsv: ImageLike) -> np.ndarray:
        if isinstance(obsv, Image.Image):
            im = obsv
        elif isinstance(obsv, (np.ndarray, tf.Tensor)):
            obsv = tf.convert_to_tensor(obsv)
            obsv = np.asarray(obsv)
            im = Image.fromarray(obsv)
        else:
            raise ValueError(
                f"Unsupported state type: `{type(obsv)}`. "
                f"Only `PIL.Image.Image`, `np.ndarray` and `tf.Tensor` "
                f"are supported."
            )

        im = im.convert("L")
        im.thumbnail((80, 125), Image.Resampling.LANCZOS)  # shape=(80, 125)

        width, height = im.size
        # Crop the image according to the (l, r, u, d) borders:
        im = im.crop((0, height - 80, width, height))  # shape=(80, 80)

        p_obsv = np.asarray(im) / 255.0  # shape=(80, 80)
        p_obsv = np.expand_dims(p_obsv, -1)  # shape=(80, 80, 1)

        return p_obsv


airraid_preprocessor = AirRaidPreprocessor()

airraid_dqn_layer_configs = (
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
    LayerConfig(Dense, dict(units=gym.make('ALE/AirRaid-v5').action_space.n,
                            activation=None, name="q_layer"))
)


def train_airraid(
    display=False,
    save_path="airraid-training-saves"
):
    if display:
        env = gym.make('ALE/AirRaid-v5', render_mode="human")
    else:
        env: EnvProtocol = gym.make('ALE/AirRaid-v5')

    run_dqn_algorithm(
        env,
        airraid_dqn_layer_configs,
        preprocess=airraid_preprocessor,
        preprocessed_obsv_shape=airraid_preprocessor.preprocessed_obsv_shape,
        load_path=save_path,
        save_path=save_path,
        num_time_steps=100000
    )
    env.close()


def play_airraid(
    play_rounds=100,
    display=False,
    load_save_path=None
):
    if display:
        env = gym.make('ALE/AirRaid-v5', render_mode="human")
    else:
        env = gym.make('ALE/AirRaid-v5')
    dqn: DQN = DQN(
        state_shape=(1, *airraid_preprocessor.preprocessed_obsv_shape),
        layer_configs=airraid_dqn_layer_configs
    )
    dqn.load(load_save_path)

    total_reward = 0.0
    for i in range(play_rounds):
        observation = env.reset()[0]
        observation = airraid_preprocessor(observation)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = dqn.select_action(np.array([observation]))
            observation, reward, terminated, truncated, _ = env.step(action)
            observation = airraid_preprocessor(observation)
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

    save_path = "airraid-training-saves"
    if mode in ("t", "train"):
        train_airraid(display=False, save_path=save_path)
    elif mode in ("p", "play"):
        play_airraid(play_rounds=5, display=True,
                     load_save_path=save_path)
    else:
        print("Invalid argument! Mode can only be 't' or 'p', but '{0}' "
              "received.".format(mode))
