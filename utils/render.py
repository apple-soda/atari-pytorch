import gym
import time
import matplotlib.pyplot as plt

def render(agent, env, path, runs, timesleep=0.05):
    
    agent = agent
    agent.load(path) # sets epsilon to 0.01
    agent.network.eval()
    
    state = env.reset()
    t_reward = 0
    done = False
    
    while not done:
        env.render()
        
        action = agent.action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        t_reward += reward
        
        time.sleep(timesleep)
