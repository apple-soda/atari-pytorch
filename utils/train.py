import gym
import numpy as np
import time
import pickle
import os

from core.replay import *
from environments.wrappers import *
from agents.dqn import *
from utils.logger import *

def standard_train(agent, env, **params):
    # training parameters
    num_steps = params['num_steps']
    total_steps = params['total_steps']
    logger = params['logger']
    save_freq = params['save_freq']
    e_verbose = params['e_verbose']
    save_dir = params['save_dir']
    
    t_reward = np.array([])
    
    start_time = time.time()
    ep = 0
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    log = {'agent':params['file_name'], 'params':params, 'episodes':[]}
    
    while num_steps < total_steps:
        done = False
        state = env.reset()
        sum_reward = 0
        while not done:
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.update()

            sum_reward += reward
            state = next_state
            num_steps += 1

        #agent.update()
        t_reward = np.append(t_reward, sum_reward)
        ep += 1
        log['episodes'] = ep
        
        if ("return" in info and logger is not None):
            logger.log(f'{num_steps}, {info["return"]}')
        
        # print training status
        if ep % e_verbose == 0:
            end_time = time.time()
            print(f'num steps : {num_steps}, chance : {sum_reward}, average reward: {np.mean(t_reward)}, memory length: {len(agent.replay)}, optim steps: {agent.num_model_updates}, time: {end_time - start_time}')
            start_time = time.time()
            t_reward = np.array([])

        # save agent progress
        if ep % save_freq == 0:                
            agent.save(save_dir + params['file_name'] + '.pth')
            with open(save_dir + params['file_name'] + '.pkl', 'wb') as f:
                pickle.dump(log, f)
            if e_verbose:
                print('Episode ' + str(ep) + ': Saved model weights and log.')

"""
environment runners are helpful for reducing memory/RAM usage 
env_runner(16) -> runs the environment for 16 steps, add MDP tuple to agent memory
perform n gradient updates (typically n = 4)

vs

standard train
run the environment -> add MDP tuple to agent memory -> perform gradient upate
"""
                
def runner_train(agent, env, **params):
    # training parameters
    num_steps = params['num_steps']
    total_steps = params['total_steps']
    steps = params['steps']
    logger = params['logger']
    save_freq = params['save_freq']
    e_verbose = params['e_verbose']
    
    # env_runner
    env_runner = Env_Runner(env, agent, logger)
    
    # other
    t_reward = np.array([])
    start_time = time.time()
    ep = 0
    
    while num_steps < total_steps:
        sum_reward = 0
        tuple_list = env_runner.run(steps)
        for i in tuple_list:
            agent.remember(i[0], i[1], i[2], i[3], i[4])
            sum_reward += i[2]
            
        agent.update(update=4)
        t_reward = np.append(t_reward, sum_reward)
        num_steps += steps
        ep += 1
        
        if ep % e_verbose == 0:
            end_time = time.time()
            print(f'num steps : {num_steps}, chance : {sum_reward}, average reward: {np.mean(t_reward)}, memory length: {len(agent.replay)}, optim steps: {agent.num_model_updates}, time: {end_time - start_time}')
            start_time = time.time()
            t_reward = np.array([])
        