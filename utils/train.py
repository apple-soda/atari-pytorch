import sys
sys.path.append('..')

import gym
import numpy as np

from core.replay import *
from environments.wrappers import *
from agents.dqn import *

def train(agent, env, episodes, e_verbose, start):
    
    t_steps = 0
    t_rewards = []
    
    for e in range(episodes):
        done = False
        sum_reward = 0
        state = env.reset()
        
        while not done:
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            if t_steps > start:
                agent.update()
            
            state = next_state
            sum_reward += reward
            t_steps += 1
            
        t_rewards.append(sum_reward)
        
        if e % e_verbose == 0:
            print(f'*** episode: {e}, average reward: {np.mean(t_rewards)}, optim steps: {agent.optim_steps}, memory: {len(agent.memory)} ***')

