import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from time import sleep

style.use("ggplot")

SIZE = 10
NUM_EPISODES = 100000

MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
E_PLAYER_REWARD = 25

epsilon = 0.90
EPS_DECAY = 0.9998

SHOW_EVERY = 20000
YES = False
#if YES:
    #start_q_table = "blob_models/qtable-1565704933.pickle" # or filename
#else:
    #start_q_table = None
LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255,175,0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x} , {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        '''
        #other way of moving, might not work
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)
        #elif choice == 4:
            #self.move(x=10, y=0)


    def move(self, x=False, y=False):
        # Letting the blob move by either allow it to make own movement or randomly updating x and y
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y

        # Making sure that blob does not go outside grid. Uses SIZE-1 because it is size 10 but goes from 0-9
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE -1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE -1


enemy_q_table = {}
player_q_table = {}
for x1 in range(-SIZE+1, SIZE):
    for y1 in range(-SIZE+1, SIZE):
        for x2 in range(-SIZE+1, SIZE):
            for y2 in range(-SIZE+1, SIZE):
                player_q_table[((x1, y1), (x2,y2))] = np.random.uniform(-5, 0, size = 4)
                enemy_q_table[((x1, y1), (x2,y2))] = np.random.uniform(-5, 0, size = 4)

player_episode_rewards = []
enemy_episode_rewards = []

def take_action(player, food, enemy):
    player_obs = (player-food, player-enemy)
    enemy_obs = (enemy-food, enemy-player)
    if np.random.random() > epsilon:
        player_action = np.argmax(player_q_table[player_obs])

    else:
        player_action = np.random.randint(0,4)
    if np.random.random() > epsilon:
        enemy_action = np.argmax(enemy_q_table[enemy_obs])

    else:
        enemy_action = np.random.randint(0,4)
    return player_action, enemy_action, player_obs, enemy_obs


for episode in range(NUM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on episode # {episode}, epsilon: {epsilon}")
        print(f"player : last {SHOW_EVERY} episodes mean {np.mean(player_episode_rewards[-SHOW_EVERY:])}")
        print(f"enemy : last {SHOW_EVERY} episodes mean {np.mean(enemy_episode_rewards[-SHOW_EVERY:])}")
        show =True
    else:
        show = False

    player_episode_reward = 0
    enemy_episode_reward = 0
    for i in range(200):
        player_action, enemy_action, player_obs, enemy_obs = take_action(player=player, food=food, enemy=enemy)
        player.action(player_action)
        enemy.action(enemy_action)
        food.move()
        # if show:
        #     sleep(0.015)

        if player.x == enemy.x and player.y == enemy.y:
            player_reward = -ENEMY_PENALTY
            enemy_reward = FOOD_REWARD
        elif player.x == food.x and player.y == food.y:
            #print("got food")
            player_reward = FOOD_REWARD
            enemy_reward = -ENEMY_PENALTY
        else:
            player_reward = -MOVE_PENALTY
            enemy_reward = -MOVE_PENALTY

        new_player_obs = (player-food, player -enemy)
        new_enemy_obs = (enemy-food, enemy-player)

        player_max_future_q = np.max(player_q_table[new_player_obs])
        enemy_max_future_q = np.max(enemy_q_table[new_enemy_obs])
        player_current_q = player_q_table[player_obs][player_action]
        enemy_current_q = enemy_q_table[enemy_obs][enemy_action]

        if player_reward == FOOD_REWARD:
            new_player_q = FOOD_REWARD
            new_enemy_q = -ENEMY_PENALTY

        elif player_reward == -ENEMY_PENALTY:
            new_player_q = -ENEMY_PENALTY
            new_enemy_q = FOOD_REWARD
        else:
            new_player_q = (1 - LEARNING_RATE) * player_current_q + LEARNING_RATE * (player_reward + DISCOUNT * player_max_future_q)
            new_enemy_q = (1 - LEARNING_RATE) * enemy_current_q + LEARNING_RATE * (enemy_reward + DISCOUNT * enemy_max_future_q)

        player_q_table[player_obs][player_action] = new_player_q
        enemy_q_table[enemy_obs][enemy_action] = new_enemy_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]
            env[player.y][player.x] = d[PLAYER_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((500,500))
            cv2.imshow("", np.array(img))
            if player_reward == FOOD_REWARD or player_reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        player_episode_reward += player_reward
        enemy_episode_reward += enemy_reward
        if player_reward == FOOD_REWARD or player_reward == -ENEMY_PENALTY:
            break
        #print(i)
        #sleep(1)
    player_episode_rewards.append(player_episode_reward)
    enemy_episode_rewards.append(enemy_episode_reward)
    epsilon *= EPS_DECAY
player_moving_avg = np.convolve(player_episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
enemy_moving_avg = np.convolve(enemy_episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot(player_moving_avg)
plt.plot(enemy_moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel("episode #")
plt.show()

#with open(f"blob_models/qtable-{int(time.time())}.pickle", "wb") as f:
    #pickle.dump(q_table, f)
