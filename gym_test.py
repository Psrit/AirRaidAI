import gym
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from dqn import DQN, LayerConfig

env = gym.make('ALE/AirRaid-v5', render_mode="human")
frame_shape = env.observation_space.shape
action_dim = env.action_space.n

USE_DQN = True

if USE_DQN:
    layer_configs = [
        LayerConfig(
            Conv2D,
            dict(filters=32, kernel_size=(8, 8), strides=4,
                 activation="relu", name="Conv1", padding="same")
        ),
        LayerConfig(
            Conv2D,
            dict(filters=64, kernel_size=(4, 4), strides=2,
                 activation="relu", name="Conv2", padding="same")
        ),
        LayerConfig(
            Conv2D,
            dict(filters=64, kernel_size=(3, 3), strides=1,
                 activation="relu", name="Conv3", padding="same")
        ),
        LayerConfig(
            Flatten, {}
        ),
        LayerConfig(
            Dense,
            dict(units=512, activation="relu", name="FullyConnected1")
        ),
        LayerConfig(
            Dense,
            dict(units=action_dim, activation="relu", name="FullyConnected2")
        )
    ]

    network = DQN(frame_shape, layer_configs,
                  frame_preprocessor=lambda x: x)

for i_episode in range(3):
    observation = env.reset()
    total_reword = 0
    for t in range(30):
        if USE_DQN:
            action = network.select_action_from_frame(observation)
        else:
            action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reword += reward
        print(t, action, reward, info)
        if done:
            print("Episode finished after {} timesteps.".format(t + 1))
            break
    print("total reward: {0}".format(total_reword))
    c = input()
