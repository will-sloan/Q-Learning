import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 20007
TALLY_EVERY = 50

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

# Part 2
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)



discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
#print(q_table.shape)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max' : []}



def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))



for episode in range(EPISODES):
    episode_reward= 0
    # When to show the visual of how the model is doing.
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    # Set to False, but is replaced later in step`
    done = False

    while not done:
        # See if random step should be taken
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        # Takes action
        new_state, reward, done, _ = env.step(action)
        # Tallies this agents lifes total reward
        episode_reward += reward
        #Chnages the observations into discrete values
        new_discrete_state = get_discrete_state(new_state)
        #Checks if this episode should be shown
        if render:
            env.render()
        # If the agent was not terminated, then calculate new q value
        # for the q table
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q  = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q
        # Checks if cart is at the flag
        elif new_state[0] >= env.goal_position:
            print(f"Go to goal at episode {episode}")
            # Sets the q value to the top value (highest reward is 0)
            q_table[discrete_state + (action, )] = 0
        # Sets the state to the new state, since the q value calculation is based on the previous observation_space
        discrete_state = new_discrete_state
    # Check if epsilon is still in range
    if  END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    # Gather total reward of the episode
    ep_rewards.append(episode_reward)
    # Same as episode % TALLY_EVERY == 0
    # If it is a time to show the gui, then make these calculations for the previous TALLY_EVERY
    # If TALLY_EVERY = 1000, then this would get the average, min  and max of the past 1000 episodes

    if not episode % TALLY_EVERY:
        average_reward = sum(ep_rewards[-TALLY_EVERY:])/len(ep_rewards[-TALLY_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-TALLY_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-TALLY_EVERY:]))

        print(f"Episode {episode} avg {average_reward} min {min(ep_rewards[-TALLY_EVERY:])} max {max(ep_rewards[-TALLY_EVERY:])}")



env.close()
# Makes plot of the aggr_ep_rewards.
# A new point is added every TALLY_EVERY. So in this case, every 50 episodes.
# To add a dot to each data point, add marker='o' like so plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg', marker='o')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
