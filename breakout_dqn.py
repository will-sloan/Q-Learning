# All Purpose DQN for atari
import cv2
import gym
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
OBS_SPACE = (84,84, 1)
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
            model.add(Conv2D(64, (3,3), input_shape=OBS_SPACE, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Conv2D(32, (3,3), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=env.action_space.n, activation='linear'))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        state = np.array(state)
        return self.model.predict(np.array(state).reshape((1,84,84,1))/255)[0]
        #return self.model.predict(np.array(state)/255)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_q_list = self.model.predict(np.expand_dims(current_states, axis = -1))
        #current_q_list = self.model.predict(current_states)
        new_current_states = (np.array([transition[3] for transition in minibatch])/255)
        future_q_list = self.backup_model.predict(np.expand_dims(new_current_states, axis=-1))
        #future_q_list = self.backup_model.predict(new_current_states)
        X = []
        y= []


        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_q_list)
                new_q = max_future_q * DISCOUNT + reward

            else:
                new_q = reward

            current_qs = current_q_list[index]
            current_qs[action] = new_q

            X.append(np.reshape(current_state, (1,84,84,1)))
            y.append(current_qs)
        self.model.fit(np.array(X).reshape(64,84,84,1)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if terminal_state:
            self.backup_update_counter +=1

        if self.backup_update_counter > UPDATE_BACKUP_EVERY:
            self.backup_update_counter.set_weights(self.model.get_weights())
            self.backup_update_counter = 0
def convert_gray(image):
    #grey =cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (84,84), interpolation=cv2.INTER_LINEAR).reshape(1,84,84,1)
    grey =cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (84,84), interpolation=cv2.INTER_LINEAR)
    #print(grey.shape)
    return grey

aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max' : []}
REPLAY_MEMORY_SIZE = 50_000
LOAD_MODEL = None
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 64
DISCOUNT = 0.95
UPDATE_BACKUP_EVERY = 5
epsilon = 1
EPSILON_DECAY = 0.998
EPISODES = 1000
SHOW = False
AGGREGATE_STATS_EVERY = 100
ep_rewards = [-200]
MIN_EPSILON = 0.01
fit_every_ep = 5
agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
    current_state = convert_gray(env.reset())
    episode_reward = 0
    done = False
    count_step = 0
    while not done:
        #print()
        #print(done)
        if count_step % fit_every_ep and SHOW: #and not episode % AGGREGATE_STATS_EVERY
            env.render()

        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, convert_gray(new_state), done))
        if count_step % fit_every_ep == 0:
            #print(f"training at step {count_step} in ep {episode}")
            agent.train(done)
        current_state = convert_gray(new_state)
        count_step += 1


    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        print(f"Episode Stats for : {episode}, avg: {average_reward} min: {min_reward} max: {max_reward}")

        agent.model.save(f'Breakout__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
