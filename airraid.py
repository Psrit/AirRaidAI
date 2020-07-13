import gym
from dqn import DQNBrain
import os, sys, getopt

NUM_EPISODES = 100  # Episode limitation
NUM_STEPS_PER_EPISODE = 300  # Step limitation in an episode

# For every `NUM_EPISODES_FOR_NEXT_TEST_ROUND` episodes, conduct one round of
# test, which includes `NUM_TESTS_PER_TEST_ROUND` tests.
# In each test, we play the game for `NUM_STEPS_PER_EPISODE` steps (if the game
# is not lost).
NUM_EPISODES_FOR_NEXT_TEST_ROUND = 5
NUM_TESTS_PER_TEST_ROUND = 5


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

    for episode in range(NUM_EPISODES):
        print("========== Episode {0} ==========".format(episode))
        observation = env.reset()
        # Go `NUM_STEPS_PER_EPISODE` steps in every episode
        for step in range(NUM_STEPS_PER_EPISODE):
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
            # If the game ends, stop training steps of this episode immediately.
            if done:
                break

        # Test `NUM_TESTS_PER_TEST_ROUND` times for every `NUM_EPISODES_FOR_NEXT_TEST_ROUND` episodes
        if episode > 0 and episode % NUM_EPISODES_FOR_NEXT_TEST_ROUND == 0:
            total_reward = 0.0
            for i in range(NUM_TESTS_PER_TEST_ROUND):
                print("[TEST] Test: " + str(i))
                observation = env.reset()
                for j in range(NUM_STEPS_PER_EPISODE):
                    # print("step: " + str(j))
                    if test_display:
                        env.render()
                    action = brain.action(observation / 255.0)
                    observation, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            aver_reward = total_reward / NUM_TESTS_PER_TEST_ROUND
            print('[TEST] After Episode {0}, evaluation average reward: {1}'
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
    PLAY_GAMES = 100
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
    for i in range(PLAY_GAMES):
        observation = env.reset()
        done = False
        print(i)
        while not done:
            if test_display:
                env.render()
            action = brain.action(observation / 255.0)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
    aver_reward = total_reward / PLAY_GAMES
    print('Average reward in {0} games: {1}'
          .format(PLAY_GAMES, aver_reward))

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
