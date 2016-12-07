import numpy as np
import gym

"""
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

    def __init__(self, env, monitor_path):
        self.env = env
        self.states = 16
        self.actions = 4
        self.table = np.zeros((self.states, self.actions))
        self.learning_rate = 0.05
        self.gamma = 0.9
        self.eps = 0.025
        self.num_episodes = 5000
        self.max_steps_per_episode = 100
        self.reward_history = []
        self.path_to_recording = './tmp/'+monitor_path
        self.env.monitor.start(self.path_to_recording)


    def train(self):
        beat_benchmark = False
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            steps = 0
            self.eps = 1.0 / (epoch + 1.0)
            while steps < self.max_steps_per_episode:
                steps += 1

                # if beat_benchmark:
                #     action = np.argmax(self.table[state])
                #     state_prime, reward, is_done, info = self.env.step(action)
                #     if is_done:
                #         break
                #     else:
                #         state = state_prime
                #         continue

                if np.random.random() < self.eps:
                    #print 'taking random action'
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.table[state])

                state_prime, reward, is_done, info = self.env.step(action)
                maxQ = np.max(self.table[state_prime])

                #print (state, action), epoch, steps,  reward, maxQ
                if is_done and (reward < 0.001):
                    reward = -1.0
                self.table[state, action] += self.learning_rate * (((reward - 0.1) + (self.gamma * maxQ)) - self.table[state, action]) #added a living penlaty of 0.05
                state = state_prime
                if is_done:
                    #self.eps = 60.0 /(epoch + 100.0)
                    break
            #renormalize the table periodically
            self.reward_history.append(0.0 if reward < 0.0 else reward)
            if sum(self.reward_history[-100:])/100.0 > 0.78:
                print 'epoch: ', epoch, 'beat benchmark', sum(self.reward_history[-100:])/100.0
                beat_benchmark = True
                self.learning_rate = 0.01

            if (epoch % 100) == 0:
                print 'epoch', epoch, sum(self.reward_history[-100:])/100.0, len(self.reward_history), self.eps
                #print self.table
        print np.round(self.table, 5)
        print np.max(self.table), np.min(self.table)
        print sum(self.reward_history[-100:])/100.0

        self.env.monitor.close()

if __name__ == "__main__":
    #solved: https://gym.openai.com/evaluations/eval_Phjwe0DpSciTixagE0yUZg
    # the above model is trained to solve the below problem
    env = gym.make('FrozenLake-v0')
    table = RLTable(env, 'FrozenLake-v0-2')
    table.train()
