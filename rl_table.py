import numpy as np
import gym



"""
Solved Frozen Lake: after 1100 steps.
Solved Taxi-v1: after 500 steps
Implementation of SARSA...


SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)

0, left
1, down
2, right
3, up

"""


class RLTable:

    def __init__(self,
                 env,
                 monitor_path,
                 step_penalty = 0.1,
                 learning_rate = 0.25,
                 gamma = 0.99,
                 eps = 0.025,
                 num_episodes = 5000
                 ):
        self.env = env
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        self.table = np.zeros((self.states, self.actions))
        self.step_penalty = step_penalty
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.num_episodes = num_episodes
        self.max_steps_per_episode = 100
        self.reward_history = []
        self.path_to_recording = './tmp/'+monitor_path
        self.env.monitor.start(self.path_to_recording, force  = True)


    def train(self):
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            steps = 0
            self.eps = 1.0 / (epoch + 1.0)
            #while steps < self.max_steps_per_episode:
            total_reward_per_episode = 0.0
            while 1:
                steps += 1

                if np.random.random() < self.eps:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.table[state])

                state_prime, reward, is_done, info = self.env.step(action)
                total_reward_per_episode += reward
                maxQ = np.max(self.table[state_prime])

                self.table[state, action] += self.learning_rate * (((reward - self.step_penalty) + (self.gamma * maxQ)) - self.table[state, action])

                state = state_prime

                if is_done:
                    break

            self.reward_history.append(total_reward_per_episode)

            if (epoch % 100) == 0:
                print 'epoch', epoch, sum(self.reward_history[-100:])/100.0, len(self.reward_history), self.eps

        print np.round(self.table, 5)
        print np.max(self.table), np.min(self.table)
        print sum(self.reward_history[-100:])/100.0

        self.env.monitor.close()

if __name__ == "__main__":
    env = gym.make('Taxi-v1')
    table = RLTable(env, 'Taxi-v1-1', step_penalty=0.0)
    table.train()
