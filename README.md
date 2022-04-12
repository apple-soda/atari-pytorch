# atari-reinforcement-learning
clean and tunable implementation of reinforcement learning algorithms to play atari games

### to do
* if current training framework works, add plot/save (pickle) function to save and watch agent play games
* just clean up code in general, deal with numpy deprecations and whatnot
* try to fix heavy cpu load and smoother/faster training process

## features
#### runner train:
* less computationally expensive
    * less gradient updates for # of frames
    * 'faster' training, but scaled proportionally
* more stability in training
#### epsilons:
* traditional epsilon (epsilon min, decay, etc)
* frame by frame epsilon calculation
    * can choose to cap epsilon at frame n, and will do the computation accordingly
    

### future features:
* plot/watch functions
* could clean up some save functions and whatnot to make training function cleaner and more organized

### specs/personal parameters:
* add computer specs here
* atari frames are very large and computationally expensive to store on RAM
* note that save a lot of memory but running a 'bare minimum' setup. add that setup. (aka no agent class, etc)
`memory_size`: 125000, maybe 150000 on a good day :)