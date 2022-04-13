import gym
import time
import matplotlib.pyplot as plt
from IPython import display

def render_agent(agent, env, path, runs, timesleep=0.0005):
    agent = agent
    env = env
    agent.load(path) # sets epsilon to 0.01
    agent.network.eval()
    
    for run in range(runs):
        state = env.reset()
        t_reward = 0
        done = False

        while not done:
            render(env)
            action = agent.action(state)
            next_state, reward, done, info, _ = env.step(action)
            state = next_state
            t_reward += reward

            time.sleep(timesleep)
            
        print(f'Reward: {t_reward}')
        
    env.close()

def render(env):
    plt.figure(0)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)