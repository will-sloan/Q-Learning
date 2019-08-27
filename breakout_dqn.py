# All Purpose DQN for atari

import gym
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tqdm import tqdm
import numpy as np
import random
import tensorflow as tf
print(tf.__version__)
env = gym.make('Breakout-v0')

env.reset()
#print(start_obs)

#print(env.action_space)
#print(env.observation_space)
'''
for i in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    print(f"Got reward {reward}")
env.close()
'''

class DQNAgent:

    def __init__(self):

        self.model = self.create_model()
        self.backup_model = self.create_model()
        self.backup_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)
        self.backup_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            model.load_model(LOAD_MODEL)
        else:
            model = Sequential()
            model.add(Conv2D(64, (3,3), input_shape=env.observation_space.shape, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Conv2D(32, (3,3), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(64))
            model.add(Dense(units=env.action_space.n, activation='linear'))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.backup_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list)
                new_q = reward + DISCOUNT * max_future_q

            else:
                new_q  = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        self.model.fit(np.array(X)/255, np.array(y), batch_size= MINIBATCH_SIZE, verbose=0, shuffle=False)

        if terminal_state:
            self.backup_update_counter += 1

        if self.backup_update_counter > UPDATE_BACKUP_EVERY:
            self.backup_model.set_weights(self.model.get_weights())
            self.backup_update_counter = 0

REPLAY_MEMORY_SIZE = 50_000
LOAD_MODEL = None
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 64
DISCOUNT = 0.95
UPDATE_BACKUP_EVERY = 5
epsilon = 1
EPSILON_DECAY = 0.998
EPISODES = 10000
SHOW = True
AGGREGATE_STATS_EVERY = 100
ep_rewards = [-200]
MIN_EPSILON = 0.01
agent = DQNAgent()
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max' : []}

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
    current_state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        if SHOW:
            env.render()

        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        print(f"done is {done}")
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)
        current_state = new_state


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
