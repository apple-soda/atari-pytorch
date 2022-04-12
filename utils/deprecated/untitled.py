'''
slow learning but works
'''

env_name = 'BreakoutNoFrameskip-v4'
#env_name = 'PongNoFrameskip-v4'
#env_name = 'SpaceInvadersNoFrameskip-v4'

# hyperparameter

num_stacked_frames = 4

replay_memory_size = 100000

lr = 2.5e-5 # SpaceInvaders #1e-4 for PONG | 2.5e-5 for Breakout
gamma = 0.99
minibatch_size = 32

final_eps_frame = 1000000
total_steps = 20000000

target_net_update = 10000 # 10000 steps

logger = Logger("training_info")
logger.log("training_step, return")

save_model_steps = 500000

# init
raw_env = gym.make(env_name)
env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=True)

# create agent
params = {'gamma':0.99, 'alpha':lr, 'memory_size':100000, 'device':'cuda:0', 'target_net_updates':10000, 'batch_size':32, 'epsilon': 1.0,
          'final_frame_eps':1000000, 'eps_interval':0.9, 'epsilon_min':0.1}

agent = DQNAgent(raw_env.observation_space, raw_env.action_space, **params)

num_steps = 0

start_time = time.time()
while num_steps < total_steps:
    
    # if doesnt work try changing this
    # get data
    state = env.reset()
    done = False
    sum_reward = 0
    agent.update_epsilon()
    
    while not done:
        
        #state = torch.tensor(state)
        action = agent.action(state)
        next_state, reward, done, info, additional_done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        num_steps += 1
        
        # LEARN
        agent.update()
            
        if num_steps % 50000 == 0:
            end_time = time.time()
            print(f'*** total steps: {num_steps} | time(50K): {end_time - start_time} ***')
            start_time = time.time()
          
    if "return" in info:
        logger.log(f'{num_steps},{info["return"]}')

    # print time
    # save the dqn after some time
    if num_steps % save_model_steps == 0:
        torch.save(agent,f"{env_name}-{num_steps}.pt")

env.close()