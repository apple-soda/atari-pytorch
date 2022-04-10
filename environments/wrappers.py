import gym
import numpy as np
import cv2
from collections import deque

class AtariWrapper(gym.Wrapper):
    def __init__(self, env, size, k):
        self.env = env
        self.k = k
        self.size = size
        self.frames = deque(maxlen=k)
        
    def reset(self):
        self.returns = 0
        state = self.env.reset()
        state = self.process(state)
        state = np.stack([state for i in range(self.k)]) 
        return state
    
    def step(self, action):
        t_reward = 0
        done = False
        
        for i in range(self.k):
            next_state, reward, d, info = self.env.step(action)
            t_reward += reward
            next_state = self.process(next_state)
            self.frames.append(next_state)
            
            if d:
                done = True
                break
        
        # tracks return for logger
        self.returns += reward
        if done:
            info["return"] = self.returns
            
        # need to figure out a way to return a shape (1, 4, 84, 84) without turning it into 5 dimensional tensor during batch training
        return np.stack(self.frames), t_reward, done, info
    
    def process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, dsize=(self.size))
        return frame  