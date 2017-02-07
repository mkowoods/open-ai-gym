
-------------
#Description
-------------

###Solutions to Open AI Gym programs using different Reinforcement Learning and Deep Learning Approaches

------------------

##Solutions with Descriptions

#####Below is a list of some of the projects in the repo

* rl_q_learning.py
 - Implementation of Q Learning Model Using Dynamic Programming, in which you have table that represents every 
 state / action pair. 
 - This model was able to solve several basic problems like FrozenLake and Taxi with minimal tuning.
 
* cartpole - policy_gradient.ipynb
 - Implementation of a Policy Gradient / Value Gradient
 - The policy gradient is meant to learn overtime which action to take given an observed state, by comparing the true
 observed reward for a given state to the expected reward. The value gradient is responsible for providing the estimate 
 for the future reward. The PG is implementing a version of logistic regression and the VG uses a 2-layer regression NN
 
* cartpole - random_search.ipynb (monte carlo)
 - Uses random search to identify a set of weights such that heaviside_step_func( np.dot(state, weights) ), is able to 
 beat the required benchmarks. This actually work pretty well for this task and you're able to quickly learn a set of weights
 that can perform as well as more advance techniques. Obviously would not scale to larger state space sizes
 
* cartpole - kmeans with q_learning.ipynb
 - attempts to use kmean clustering to decrease the size of the state space so that a q learning model can be applied
 - this was particularly successful
 
* cartpole - dqn.ipynb
 - implement a version of deep q learning with experience replay
 - the network takes an input of 

#References

https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

https://github.com/dennybritz/reinforcement-learning

http://kvfrans.com/simple-algoritms-for-solving-cartpole/

Neural Evolution:
https://gist.github.com/DollarAkshay/14059981d90c98607339d3ee17d2f0e9#file-openai_cartpole_v0-py
