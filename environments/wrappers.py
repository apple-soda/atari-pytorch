import gym
import numpy as np
import cv2
from collections import deque

# future todo: add additional done for games with lives
class AtariWrapper(gym.Wrapper):
    def __init__(self, env, k, img_size=(84,84), use_add_done=False):
        super(AtariWrapper, self).__init__(env)
        self.img_size = img_size
        self.k = k
        self.use_add_done = use_add_done
        self.frame_stack = deque(maxlen=k)
        
    def reset(self):
        self.last_life_count = 0
        ob = self.env.reset()
        ob = self.preprocess_observation(ob)
        for i in range(self.k):
            self.frame_stack.append(ob)
        return np.stack(self.frame_stack) # [4, 84, 84]
    
    def step(self, action): 
        reward = 0
        done = False
        additional_done = False
        
        # k frame skips or end of episode
        for i in range(self.k):
            ob, r, d, info = self.env.step(action)
    
            # insert a (additional) done, when agent loses a life (Games with lives)
            if self.use_add_done:
                if info['ale.lives'] < self.last_life_count:
                    additional_done = True  
                self.last_life_count = info['ale.lives']
            
            ob = self.preprocess_observation(ob)
            self.frame_stack.append(ob)
            reward += r
            
            if d: # env done
                done = True
                break
        return np.stack(self.frame_stack), reward, done, info, additional_done # [4, 84, 84]
      
    def preprocess_observation(self, ob):
    # resize and grey and cutout image
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = cv2.resize(ob, dsize=self.img_size)
        return ob