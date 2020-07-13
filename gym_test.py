import gym
from airraid import NUM_EPISODES, NUM_STEPS_PER_EPISODE

env = gym.make('AirRaid-v0')
print(type(env))
for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    total_reword = 0
    for t in range(NUM_STEPS_PER_EPISODE):
        env.render()
        # print(observation.shape)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reword += reward
        print(t, action, reward, info)
        if done:
            print("Episode finished after {} timesteps.".format(t + 1))
            break
    print("total reward: {0}".format(total_reword))
    c = input()
