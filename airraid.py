import gym
from dqn import DQNBrain
import os, sys, getopt

EPISODE = 100  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST_NUM = 5  # The number of experiment test every `TEST_STEP` episode
TEST_STEP = 5
PLAY_STEP = 100


def train_air_raid(train_display=False, test_display=True,
                   record_test=True, record_path="records",
                   record_cost=False):
    env = gym.make('AirRaid-v0')
    # env = gym.wrappers.Monitor(env, './airraid-experiment')

    # init DQNBrain
    brain = DQNBrain(env, pooling_scale=4, record=record_cost)  # , [8, 4], [16, 32], 256, 4)
    brain.add_conv("conv1", patch_size=4, depth=16, pooling=True)
    brain.add_conv("conv2", patch_size=4, depth=32, pooling=True)
    brain.add_fc("fc1", num_nodes=256, )
    brain.add_fc("q_layer", num_nodes=-1, activation=None, output_layer=True)

    # init Q network
    brain.initialize_network()

    for episode in range(EPISODE):
        print("========== Episode {0} ==========".format(episode))
        observation = env.reset()
        # Go `STEP` steps in every episode
        for step in range(STEP):
            # print("step: " + str(step))
            if train_display:
                env.render()
            action = brain.epsilon_greedy_action(observation / 255.0)
            new_observation, reward, done, _ = env.step(action)
            # print new_observation, reward, action
            brain.perceive(
                observation / 255.0, action, reward / 100.0, new_observation / 255.0, done
            )
            observation = new_observation
            # If the game ends, stop training steps of this round immediately.
            if done:
                break

        # Test `TEST_NUM` times for every `TEST_STEP` episodes
        if episode > 0 and episode % TEST_STEP == 0:
            total_reward = 0.0
            for i in range(TEST_NUM):
                print("[TEST] Test round: " + str(i))
                observation = env.reset()
                for j in range(STEP):
                    # print("step: " + str(j))
                    if test_display:
                        env.render()
                    action = brain.action(observation / 255.0)
                    observation, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            aver_reward = total_reward / TEST_NUM
            print('[TEST] Episode: {0} Evaluation average reward: {1}'
                  .format(episode, aver_reward))

            if record_test:
                if not record_path.endswith(os.sep):
                    record_path += os.sep
                if not os.path.isdir(record_path):
                    os.makedirs(record_path)
                record_file = record_path + "test_rewards"
                rf = open(record_file, "a")
                try:
                    rf.write(str(aver_reward) + "\n")
                finally:
                    rf.close()

    env.close()


def play_air_raid(test_display=False):
    import time
    time_str = time.strftime("%Y%m%d_%H%M%S")

    env = gym.make('AirRaid-v0')
    env = gym.wrappers.Monitor(env, './play_videos/' + time_str)

    # init DQNBrain
    brain = DQNBrain(env, pooling_scale=4, record=False)  # , [8, 4], [16, 32], 256, 4)
    brain.add_conv("conv1", patch_size=4, depth=16, pooling=True)
    brain.add_conv("conv2", patch_size=4, depth=32, pooling=True)
    brain.add_fc("fc1", num_nodes=256, )
    brain.add_fc("q_layer", num_nodes=-1, activation=None, output_layer=True)

    # init Q network
    brain.initialize_network()

    total_reward = 0.0
    for i in range(PLAY_STEP):
        observation = env.reset()
        done = False
        print(i)
        while not done:
            if test_display:
                env.render()
            action = brain.action(observation / 255.0)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
    aver_reward = total_reward / PLAY_STEP
    print('Average reward in {0} rounds: {1}'
          .format(PLAY_STEP, aver_reward))

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
        train_air_raid(train_display=False, test_display=True,
                       record_test=True, record_cost=False)
    elif mode in ("p", "play"):
        play_air_raid()
    else:
        print("Invalid argument! Mode can only be 't' or 'p', but '{0}' "
              "received.".format(mode))
