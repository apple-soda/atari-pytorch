class Env_Runner:
    def __init__(self, env, agent, logger):
        super().__init__()
        self.env = env
        self.agent = agent
        
        self.logger = logger
        self.logger.log("training_step, return")
        
        self.ob = self.env.reset()
        self.total_steps = 0
        
    def run(self, steps):
        obs = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(steps):
            action = self.agent.action(self.ob)
            obs.append(self.ob)
            actions.append(action)
            self.ob, r, done, info = self.env.step(action)
               
            if done: 
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{self.total_steps+step}, {info["return"]}')
            
            rewards.append(r)
            dones.append(done)
            
        self.total_steps += steps
                                    
        tuples = self.transition(obs, actions, rewards, dones)
        return tuples
    
    def transition(self, obs, actions, rewards, dones):
        tuples = []
        steps = len(obs) - 1
        
        for t in range(steps):
            tuples.append((obs[t], actions[t], rewards[t], obs[t+1], int(not dones[t])))

        return tuples