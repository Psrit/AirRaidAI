# parameters for ReplayMemory
REPLAY_CAPACITY = 320

# parameters for DQN
BATCH_SIZE = 64
TIME_STEPS_PER_TRAIN = 100  # number of `time_step`s per train once after perceiving
# Auto synchronize Q model weights after training
# if train_step % TRAIN_STEPS_PER_Q_SYNC == 0.
TRAIN_STEPS_PER_Q_SYNC = 15


# parameters for main-loop of AirRaid training
NUM_EPISODES = 30  # Episode limitation
NUM_STEPS_PER_EPISODE = 300  # Step limitation in an episode
NUM_EPISODES_FOR_NEXT_SAVE = 1  # number of episodes per model saving
