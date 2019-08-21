
import numpy as np
import cv2
import random
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt


def collision_with_boundaries(snake_head):
    #print(f"Head of snake is at {snake_head[0]} and {snake_head[1]}")
    if snake_head[0] >= SIZE or snake_head[0] <0 or snake_head[1] >= SIZE or snake_head[1] < 0:

        #print('you are out')
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = copy.deepcopy(snake_position[0])
    #print(f'Check if Snake Head {snake_head} is in snake_position {snake_position[1:]} \n')
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

def collision_with_apple(apple_position, score):
    apple_position = np.random.randint(0, SIZE-1, size=2)
    score +=1
    return apple_position, score

def move(k, prev_button_direction, button_direction):
    #time.sleep(1)
    if k == 1 and prev_button_direction != 1:
       button_direction = 0
    elif k == 0 and prev_button_direction != 0:
       button_direction = 1
    elif k == 2 and prev_button_direction != 2:
       button_direction = 3
    elif k ==3 and prev_button_direction != 3:
       button_direction = 2
    else:
       button_direction = button_direction
    prev_button_direction = button_direction
    #print(f"Taking action {str(k)}")
    #print(f"Midway through action, Snake position is {snake_position}")
    # Change the head position based on the button direction
    if button_direction == 1:

       snake_head[0] += 1
       #print(f"At action 1, Snake position is {snake_position}")
    elif button_direction == 0:
       snake_head[0] -= 1
    elif button_direction == 2:
       snake_head[1] += 1
    elif button_direction == 3:
       snake_head[1] -= 1
    #print(f"At end of action, Snake position is {snake_position}")

    #print(f"new button direction is {button_direction}")
    return snake_head, prev_button_direction, button_direction


DISCOUNT = 0.99
LEARNING_RATE = 0.8
NUM_ESPIODES = 100000

SHOW_EVERY = 10000
TALLY_EVERY = 10000

STEP_PENALTY = 1
LOSS_PENALTY = 100
FOOD_REWARD = 10


SIZE= 20

epsilon = 0.5
EPS_DECAY = 0.99998

start_q_table = None

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE+1):
        for y1 in range(-SIZE+1, SIZE+1):
                q_table[((x1, y1))] = np.random.uniform(-5, 0, size = 4)


#img = np.zeros((200,200, 3), dtype='uint8')

#score = 0
#count = 0
score_tally = []
episode_rewards = []
for episode in range(NUM_ESPIODES):
    button_direction = 1
    prev_button_direction = 1
    snake_head = np.random.randint(0, SIZE-1, size=2)
    snake_position = [[snake_head[0],snake_head[1]], [snake_head[0]+1, snake_head[1]], [snake_head[0]+2, snake_head[1]]]


    # Display apple (Red rectangles)
    apple_position = np.random.randint(0,SIZE-1, size = 2)
    score = 0
    episode_reward = 0


    for i in range(1000):
        reward = 0
        #print(f"At start of while, Snake action is {button_direction}")
        #print(f"At start of while, Snake position is {snake_position}")
        #print("before insert", snake_position, "list head", list(snake_head), "nonlist", snake_head)


        #print("before insert", snake_position, "list head", list(snake_head), "nonlist", snake_head)


        # t_end = time.time() +0.2
        # k = None
        # while time.time() < t_end:
        #     if k == None:
        #         k = cv2.waitKey(10)
        #     else:
        #         continue

        obs = tuple((np.array(snake_head)-np.array(apple_position)))
        #print(f"Obs is {obs} and its type {type(obs)}\n")
        #print(f"q-table is {q_table.keys()}")
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
        #print(f"Before action, Snake action is {button_direction}")
        snake_head, prev_button_direction, button_direction = move(action, prev_button_direction, button_direction)

        #print(f"Loop number: {count}")
        #print(f"Snake action is {button_direction}")

        #count +=1
        #prev_snake_position = copy.deepcopy(snake_position)
        #print(f"Snake head {type(snake_head)}\n\n")
        #print(f"apple position {type(apple_position)}\n\n")
        if all(snake_head == apple_position):
            apple_position, score = collision_with_apple(apple_position, score)
            reward = FOOD_REWARD
            snake_position.insert(0, list(snake_head))
        else:
            reward = -STEP_PENALTY
            snake_position.insert(0, list(snake_head))
            snake_position.pop()
        #print(f"After inserting {snake_head} into prev_snake_position and remove {a}, new list is {snake_position}\n")
        #prev_snake_position = copy.deepcopy(snake_position)
        if collision_with_self(snake_position) == 1 or collision_with_boundaries(snake_head) ==1:
            #print("outta here")\mx
            reward = -LOSS_PENALTY
            #print(f"Got reward for hitting self: {collision_with_self(snake_position) == 1} wall: {collision_with_boundaries(snake_head) ==1}")


        new_obs = tuple(np.array(snake_head)-np.array(apple_position))
        #print(f"snake_head is {snake_head} and apple_position is {apple_position}")
        #print(f"Action is {action} \n and obs is {obs}\n and new_obs is {new_obs}")
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -LOSS_PENALTY:
            new_q = -LOSS_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward +DISCOUNT * max_future_q)

        q_table[obs][action] = new_q
        episode_reward += reward
        #print(f"Should it be exitting reward: {reward} loss: {LOSS_PENALTY} is { reward == -LOSS_PENALTY}")
        if reward == -LOSS_PENALTY:
            #print("Tryna leave")
            break
        if not episode % SHOW_EVERY:
            time.sleep(0.05)
            env = np.zeros((SIZE,SIZE, 3), dtype=np.uint8)
            for position in snake_position:
                env[position[0]][position[1]] = (0,255,0)
            env[apple_position[0]][apple_position[1]] = (0,0,255)

            img = Image.fromarray(env, "RGB")
            img = img.resize((500,500))
            cv2.imshow("", np.array(img))
            if reward == FOOD_REWARD or reward == -LOSS_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if not episode % TALLY_EVERY:
        avg_tally = np.mean(score_tally[-TALLY_EVERY:])
        try:
            print(f"reward of {episode_reward}, score of {score}, epsilon of {epsilon}, at episode {episode}, avg_tally is {avg_tally}, avg_reward is {np.mean(episode_rewards[-TALLY_EVERY:])}, highest score of {np.max(score_tally)}")
        except:
            pass
    score_tally.append(score)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
score_avg = np.convolve(score_tally, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot(moving_avg)
plt.plot(score_avg)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel("episode #")
plt.show()
