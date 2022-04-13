import gym
import numpy as np
import time
import pickle
import os

from utils.replay import *
from environments.wrappers import *
from agents.ddqn import *
from utils.logger import *
from utils.runner import *

"""
standard training loop for online reinforcement learning agents
reset state -> take action -> env.step -> add MDP tuple to agent memory -> update gradient -> repeat
"""
def standard_train(agent, env, **params):
    # training parameters
    total_steps = params['total_steps']
    logger = params['logger']
    save_freq = params['save_freq']
    e_verbose = params['e_verbose']
    save_dir = params['save_dir']
    
    start_time = time.time()
    t_reward = []
    num_steps = 0
    ep = 0
    
    # creates new save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger.log("training_step, return")
    log = {'agent':params['file_name'], 'params':params, 'episodes':[]}
    
    while num_steps < total_steps:
        done = False
        state = env.reset()
        sum_reward = 0
        agent.update_epsilon() # if using by-frame epsilon updates
        
        while not done:
            action = agent.action(state)
            next_state, reward, done, info, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            sum_reward += reward
            state = next_state
            num_steps += 1
            
            agent.update()
            
            # print training status
            if num_steps % e_verbose == 0:
                end_time = time.time()
                print(f'Steps : {num_steps}, Average Reward: {np.mean(t_reward)}, Memory Length: {len(agent.replay)}, Optimizer Steps: {agent.num_model_updates}, Time Elapsed: {end_time - start_time}, Target Q Updates: {agent.updates}')
                start_time = time.time()
                t_reward = []

            # save agent progress
            if num_steps % save_freq == 0:                
                agent.save(save_dir + params['file_name'] + '.pth')
                with open(save_dir + params['file_name'] + '.pkl', 'wb') as f:
                    pickle.dump(log, f)
                if e_verbose:
                    print('Episode ' + str(ep) + ': Saved model weights and log.')

        t_reward.append(sum_reward)
        ep += 1
        log['episodes'] = ep
        
        if logger:
            logger.log(f'{num_steps}, {sum_reward}')
    
    env.close()

                
"""
environment runners are helpful for reducing memory/RAM usage 
env_runner(16) -> runs the environment for 16 steps, add MDP tuple to agent memory
perform n gradient updates (typically n = 4)
* still a work in progress
"""                
def runner_train(agent, env, **params):
    # training parameters
    num_steps = params['num_steps']
    total_steps = params['total_steps']
    steps = params['steps']
    logger = params['logger']
    t_save_freq = params['t_save_freq']
    t_verbose = params['t_verbose']
    save_dir = params['save_dir']
    
    # env_runner (env_runner creates logger in __init__)
    env_runner = Env_Runner(env, agent, logger)
    log = {'agent':params['file_name'], 'params':params, 'episodes':[]}

    # other
    t_reward = np.array([])
    start_time = time.time()
    ep = 0
    
    while num_steps < total_steps:
        sum_reward = 0
        tuple_list = env_runner.run(steps)
        
        # unpack list of tuples and insert into replay buffer
        # this part could be a lot cleaner, todo
        for i in tuple_list:
            agent.remember(i[0], i[1], i[2], i[3], i[4])
            sum_reward += i[2]
            
        agent.update(update=4)
        t_reward = np.append(t_reward, sum_reward)
        num_steps += steps
        ep += 1
        
        if num_steps % t_verbose < steps:
            end_time = time.time()
            print(f'Steps : {num_steps}, Average Reward: {np.mean(t_reward)}, Memory Length: {len(agent.replay)}, Optimizer Steps: {agent.num_model_updates}, Time Elapsed: {end_time - start_time}')
            start_time = time.time()
            t_reward = np.array([])
        
        if num_steps % t_save_freq < steps:                
            agent.save(save_dir + params['file_name'] + '.pth')
            with open(save_dir + params['file_name'] + '.pkl', 'wb') as f:
                pickle.dump(log, f)
            if t_verbose:
                print('Episode ' + str(ep) + ': Saved model weights and log.')
    
    env.close()
        