import numpy as np
import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import copy

LOAD_MODEL = "first_snake__-213.00max_-299.68avg_-312.00min__1566648165.model"

# Snake game that learns that  going backwards is bad
class Apple:
    def __init__(self, size):
        self.size = size
        self.x, self.y = np.random.randint(0, self.size, size=2)

    def eaten(self):
        self.x, self.y = np.random.randint(0, self.size, size=2)

class Snake:
    def __init__(self, size):
        #self.score = 0
        self.size = size
        self.prev_choice = 0
        self.x, self.y = 4,4
        self.body = [[self.x, self.y], [self.x+1,self.y], [self.x+2,self.y]]

    def action(self, apple, choice=False):
        if choice == False:
            choice = np.random.randint(0,3)
        if choice == 0:
            choice = 0
        elif choice == 1:
            choice = 1
        elif choice == 2:
            choice = 2
        elif choice == 3:
            choice = 3
        else:
            choice = choice
        self.prev_choice = choice

        if choice == 0:
            self.x += 1
        elif choice == 1:
            self.x -= 1
        elif choice == 2:
            self.y += 1
        elif choice == 3:
            self.y -= 1

        if self.hit_food(apple):
            self.body.insert(0, [self.x, self.y])
        else:
            self.body.insert(0, [self.x, self.y])
            self.body.pop()

    def hit_wall(self, size):
        #[0] --> up(low) down(high) [1] --> left(low) right(high)
        return self.x >= size or self.y >= size or self.x < 0 or self.y < 0

    def hit_self(self):
        return [self.x, self.y] in self.body[1:]

    def hit_food(self, food):
        return self.x == food.x and self.y == food.y


    def __str__(self):
        return f"{[self.x,self.y]}, {self.body}"

class Game:
    SIZE  = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    FOOD_REWARD = 50
    LOSS_PENALTY = 300
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 4
    SNAKE_N = 0
    FOOD_N = 1
    d = {
    0: (0,255,0),
    1: (0,0,255)
    }

    def reset(self):
        self.snake = Snake(self.SIZE)
        self.food = Apple(self.SIZE)

        while self.food.x == self.snake.x and self.food.y == self.snake.y:
            self.food = Apple(self.SIZE)

        self.episode_step = 0
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.snake.x-self.food.x) + (self.snake.x-self.food.y)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.snake.action(choice = action, apple = self.food)

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.snake.x-self.food.x) + (self.snake.x-self.food.y)

        if self.snake.hit_wall(self.snake.size) or self.snake.hit_self():
            reward = -self.LOSS_PENALTY
        elif self.snake.hit_food(self.food):
            self.food.eaten()
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if self.snake.hit_wall(self.SIZE) or self.snake.hit_self():
            done = True

        return new_observation, reward, done

    def render(self):
        time.sleep(1)
        img = self.get_image()
        img = img.resize((500,500))
        cv2.imshow("snake", np.array(img))
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        for part in self.snake.body:
            if part[0] >= self.SIZE or part[1] >= self.SIZE:
                continue
            env[part[0]][part[1]] = self.d[self.SNAKE_N]  # sets the food location tile to green color
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the enemy location to red
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img

class DQNAgent:
    def __init__(self):

        # gets trained
        self.model = self.create_model()
        # gets predict against
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)
        #self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0


    def create_model(self):
        if LOAD_MODEL is not None:
            model = load_model(LOAD_MODEL)
        else:
            model = Sequential()
            model.add(Conv2D(256, (3,3), input_shape= env.OBSERVATION_SPACE_VALUES, activation='relu'))
            #model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Conv2D(256, (3,3), activation='relu'))
            #model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(64))
            model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        #print(f"Return value from get_qs is {self.model.predict(np.array(state).reshape(-1, *state.shape)/255)}")
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

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
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max' : []}

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'first_snake'
MIN_REWARD = -200  # For model save
AVG_REWARD = 10
MEMORY_FRACTION = 0.20

EPISODES = 10000

epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 100
SHOW_PREVIEW = True
#prev_action = 1
env = Game()

ep_rewards = [-200]

if not os.path.isdir('snake_models'):
    os.makedirs('snake_models')

agent = DQNAgent()


for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
    if episode == 5000:
        epsilon = 1
        print("Putting epsilon back up")
    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False
    score = 0
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        prev_action = copy.deepcopy(action)
        new_state, reward, done = env.step(action)

        episode_reward += reward
        if reward == Game.FOOD_REWARD:
            score +=1

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)


        current_state = new_state
        step +=1

    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 10:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-AGGREGATE_STATS_EVERY:]))


        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= AVG_REWARD or episode == 1000:
            agent.model.save(f'{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    #print(score)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
