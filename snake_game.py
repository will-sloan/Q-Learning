
import numpy as np
import cv2
import random
import time
import copy



def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] <0 or snake_head[1] >= 500 or snake_head[1] < 0:
        print('you are out')
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
    apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
    score +=1
    return apple_position, score

def move(k, prev_button_direction, button_direction, active_log):
    #time.sleep(1)
    if k == ord('a') and prev_button_direction != 1:
       button_direction = 0
    elif k == ord('d') and prev_button_direction != 0:
       button_direction = 1
    elif k == ord('w') and prev_button_direction != 2:
       button_direction = 3
    elif k == ord('s') and prev_button_direction != 3:
       button_direction = 2
    elif k == ord('q'):
       global flag
       flag = False
    else:
       button_direction = button_direction
    prev_button_direction = button_direction
    if k == -1:
        print(k)
        active_log.append(k)
    else:
        print(chr(k))
        action_log.append(chr(k))
    #print(f"Taking action {str(k)}")
    #print(f"Midway through action, Snake position is {snake_position}")
    # Change the head position based on the button direction
    if button_direction == 1:

       snake_head[0] += 10
       #print(f"At action 1, Snake position is {snake_position}")
    elif button_direction == 0:
       snake_head[0] -= 10
    elif button_direction == 2:
       snake_head[1] += 10
    elif button_direction == 3:
       snake_head[1] -= 10
    #print(f"At end of action, Snake position is {snake_position}")

    #print(f"new button direction is {button_direction}")
    return snake_head, prev_button_direction, button_direction



action_log = []
img = np.zeros((500,500, 3), dtype='uint8')
global flag
flag = True
button_direction = 1
prev_button_direction = 1
score = 0
count = 0
snake_position = [[250,250], [240,250], [230,250]]
snake_head = [250,250]

# Display apple (Red rectangles)
apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]

while flag:
    #print(f"At start of while, Snake action is {button_direction}")
    #print(f"At start of while, Snake position is {snake_position}")
    #print("before insert", snake_position, "list head", list(snake_head), "nonlist", snake_head)
    cv2.imshow('snake', img)
    cv2.waitKey(1)
    img = np.zeros((500,500, 3), dtype='uint8')
    for position in snake_position:
        cv2.rectangle(img, (position[0], position[1]), (position[0]+10, position[1]+10), (0,255,0), 3)
    cv2.rectangle(img,(apple_position[0],apple_position[1]),(apple_position[0]+10,apple_position[1]+10),(0,0,255),3)
    #print("before insert", snake_position, "list head", list(snake_head), "nonlist", snake_head)


    t_end = time.time() +0.2
    k = None
    while time.time() < t_end:
        if k == None:
            k = cv2.waitKey(200)
        else:
            continue
    #print(f"Before action, Snake action is {button_direction}")
    snake_head, prev_button_direction, button_direction = move(k, prev_button_direction, button_direction, action_log)

    #print(f"Loop number: {count}")
    #print(f"Snake action is {button_direction}")

    count +=1
    #prev_snake_position = copy.deepcopy(snake_position)
    if snake_head == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0, list(snake_head))
    else:
        snake_position.insert(0, list(snake_head))
        a = snake_position.pop()
    #print(f"After inserting {snake_head} into prev_snake_position and remove {a}, new list is {snake_position}\n")
    #prev_snake_position = copy.deepcopy(snake_position)
    if collision_with_self(snake_position) == 1 or collision_with_boundaries(snake_head) ==1:
        #print("outta here")
        break

print(action_log)
cv2.destroyAllWindows()
'''

img = np.zeros((500,500,3),dtype='uint8')
# Initial Snake and Apple position
snake_position = [[250,250],[240,250],[230,250]]
apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
score = 0
prev_button_direction = 1
button_direction = 1
snake_head = [250,250]
while True:
    cv2.imshow('a',img)
    cv2.waitKey(1)
    img = np.zeros((500,500,3),dtype='uint8')
    # Display Apple
    cv2.rectangle(img,(apple_position[0],apple_position[1]),(apple_position[0]+10,apple_position[1]+10),(0,0,255),3)
    # Display Snake
    for position in snake_position:
        cv2.rectangle(img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)

    # Takes step after fixed time
    t_end = time.time() + 0.2
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(125)
        else:
            continue

    # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
    # a-Left, d-Right, w-Up, s-Down

    if k == ord('a') and prev_button_direction != 1:
        button_direction = 0
    elif k == ord('d') and prev_button_direction != 0:
        button_direction = 1
    elif k == ord('w') and prev_button_direction != 2:
        button_direction = 3
    elif k == ord('s') and prev_button_direction != 3:
        button_direction = 2
    elif k == ord('q'):
        break
    else:
        button_direction = button_direction
    prev_button_direction = button_direction

    # Change the head position based on the button direction
    if button_direction == 1:
        snake_head[0] += 10
    elif button_direction == 0:
        snake_head[0] -= 10
    elif button_direction == 2:
        snake_head[1] += 10
    elif button_direction == 3:
        snake_head[1] -= 10

    # Increase Snake length on eating apple
    if snake_head == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0,list(snake_head))

    else:
        snake_position.insert(0,list(snake_head))
        snake_position.pop()

    # On collision kill the snake and print the score
    if collision_with_boundaries(snake_head) == 1 or collision_with_self(snake_position) == 1:
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((500,500,3),dtype='uint8')
        cv2.putText(img,'Your Score is {}'.format(score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('a',img)
        cv2.waitKey(0)
        cv2.imwrite('D:/downloads/ii.jpg',img)
        break

cv2.destroyAllWindows()
'''
