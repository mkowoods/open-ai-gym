import numpy as np
import gym

class MDP:

    def __init__(self, env, monitor_path, force = True):
        self.env = env
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        self.num_episodes = 2500
        self.max_steps_per_episodes = 100


        self.env.monitor.start(monitor_path, force = force)

    def train(self):
        for epoch in range(self.num_episodes):
            state = self.env.reset()


if __name__ == "__main__":
    env = gym.make('Taxi-v1')
    table = MDP(env, './tmp/Taxi-v1-1')
    table.train()
