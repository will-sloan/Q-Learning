'''
import gym

env = gym.make('MountainCar-v0')
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.3
print(len(env.observation_space.high))
'''
'''
for _ in range(1000):
    env.render()
    env.step(0)
env.close()
'''
import gym
import time
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f"The observation {observation} has reward {reward}")
        time.sleep(10)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
