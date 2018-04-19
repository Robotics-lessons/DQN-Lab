from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
print(display.start())

import gym
import numpy as np
from tqdm import tqdm

# visualization helpers
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

# Set logger level
import logging
logging.basicConfig(level=logging.ERROR)


# Create  environment of CartPole-v0
env = gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

observation = env.reset()
# env.render(mode='rgb_array', close=True)
print('initial observation:', observation)

action = env.action_space.sample() # Select random action
print('random action:', action)
observation, reward, is_finished, info = env.step(action)
print('next observation:', observation)
print('reward:', reward)
print('is_finished:', is_finished)
print('info:', info)

# copied from gym examples - a model for our agent
class RandAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done=None, mode=None):
        return self.action_space.sample()
    
    def init_episode(self, observation):
        # provided for compatibility with general learner
        return self.action_space.sample()


# This learner provides a general interface for agents and environments 
# optionally visualize within a Jupyter Notebook when visualize_plt is True
def learner(agent=None, env_id='CartPole-v0', episodes=100, max_length = 100, init_reward=0, 
            ignore_done=False, visualize_plt=True, mode=None):
    # load the environment 
    env = gym.make(env_id)
    # set the agent to random if none provided
    if agent is None:
        agent = RandAgent(env.action_space)

    # each episode runs until it is observed as finished, or exceeds max_length in time steps
    episode_count = episodes
    done = False
    n_steps = np.zeros((episode_count,))

    # run the episodes - use tqdm to track in the notebook
    for i in tqdm(range(episode_count), disable=visualize_plt):
        
        # Initialize environment for each episode
        ob = env.reset()  
        reward = init_reward
        if visualize_plt:
            img = plt.imshow(env.render(mode='rgb_array')) # only call this once, only for jupyter
        
        # initialize the agent
        agent.init_episode(ob)
        n_steps[i]=max_length
        
        # run the steps in each epsisode
        for t in range(max_length):
            # render the environment
            if visualize_plt:
                img.set_data(env.render(mode='rgb_array')) # just update the data
                plt.axis('off')
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                env.render()
            
            # get agent's action
            action = agent.act(ob, reward, mode=mode)
            # take the action and get reward and updated observation
            ob, reward, done, _ = env.step(action)
            
            # terminate the steps if the problem is done
            if done and not ignore_done:
                n_steps[i] = t
                break
            if done and ignore_done and env_id=='MountainCar-v0' and ob[0]>= 0.5:
                # special case MountainCar
                # if have achieved the goal, then quit but otherwise keep going
                print("Episode {} done at step {}".format(i,t))
                print("Observations {}, Reward {}".format(ob, reward))
                n_steps[i] = t
                break
    env.render(close=True)
    return n_steps  # stats


num_episodes = 10
max_length = 200
steps = learner(episodes=num_episodes, max_length=max_length)

print("Minimum step count in {} episodes: {}".format(num_episodes, np.min(steps)))
print("Average step count in {} episodes: {}".format(num_episodes, np.mean(steps)))
print("Maximum step count in {} episodes: {}".format(num_episodes, np.max(steps)))

