import gym
import tensorflow as tf
import numpy as np
import json
import random
import time

random.seed(42)

"""
Implements a feedforward network (with one hidden layer)  with experience replace i.e. a DQN
"""

#TODO: add experience replay to the model to train on de-correlated examples
#TODO: add one hidden layer to the network
#TODO: write code to persist the models and recall them. Log test resutsl for determining which architecture are performing well



class TensorFlowFeedForwardModel:
    def __init__(self, num_input_nodes, num_hidden_layer_nodes, num_output_nodes, learning_rate):
        self.num_input_nodes = num_input_nodes
        self.num_hidden_layer_nodes = num_hidden_layer_nodes
        self.num_output_nodes = num_output_nodes
        self.learning_rate = learning_rate
        self.num_epochs = tf.Variable(0) #used to keep track of how many times the model has been saved

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=(None, self.num_input_nodes)) #state observation
        self.Q = tf.placeholder(dtype=tf.float32, shape=(None, self.num_output_nodes))          #observerd values

        self.w1 = tf.Variable(tf.random_uniform(shape = (self.num_input_nodes, self.num_hidden_layer_nodes), minval = -0.01, maxval = 0.01))
        self.b1 = tf.Variable(tf.random_uniform(shape = (self.num_hidden_layer_nodes, ), minval = -0.01, maxval = 0.01))

        self.w2 = tf.Variable(tf.random_uniform(shape = (self.num_hidden_layer_nodes, self.num_hidden_layer_nodes), minval = -0.01, maxval = 0.01))
        self.b2 = tf.Variable(tf.random_uniform(shape=(self.num_hidden_layer_nodes, ), minval=-0.01, maxval=0.01))

        self.w3 = tf.Variable(tf.random_uniform(shape = (self.num_hidden_layer_nodes, self.num_output_nodes), minval = -0.01, maxval = 0.01))
        self.b3 = tf.Variable(tf.random_uniform(shape=(self.num_output_nodes, ), minval=-0.01, maxval=0.01))

        self.hidden = tf.nn.tanh(tf.matmul(self.input_layer, self.w1) + self.b1)
        self.hidden_1 = tf.nn.tanh(tf.matmul(self.hidden, self.w2) + self.b2) #predicted output values
        self.Q_ = (tf.matmul(self.hidden_1, self.w3) + self.b3) #predicted output values

        self.predicted_action = tf.argmax(self.Q_, 1)

        self.loss = tf.reduce_mean(tf.square(self.Q - self.Q_))
        # Reguralization
        for w in [self.w1, self.w2, self.w3]:
            self.loss += 0.001 * tf.reduce_sum(tf.square(w))
        self.train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.update = self.train.minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()




class FeedForwardBrain:

    def __init__(self, env):
        self.env = env
        self.state_space_size = env.observation_space.shape[0] #if type is box
        self.action_space_size = env.action_space.n
        self.learning_rate = 0.0001
        self.gamma = 0.9
        self.RANDOM_ACTION = 0.5
        self.RANDOM_ACTION_DECAY = 0.99
        self.num_episodes = 3000
        self.max_steps_per_episode = 100
        self.MAX_MEMORY = 10000
        self.MIN_MEMORY = 500
        self.BATCH_SIZE = 16
        self.replay_memory = []
        self.num_hidden_units = 6
        self.reward_history = []
        self.step_history = []
        self.model = TensorFlowFeedForwardModel(
                                                num_input_nodes = self.state_space_size,
                                                num_hidden_layer_nodes = self.num_hidden_units,
                                                num_output_nodes = self.action_space_size,
                                                learning_rate = self.learning_rate
                                                )

    def one_hot_encode_state(self,obs):
        """
        observation is an integer representing a tile on the grid
        the encoding is 1 hot encoding where the array is size of state_space
        :param obs:
        :return:
        """
        identity_matrix = np.identity(self.state_space_size)
        return identity_matrix[obs: (obs + 1)]

    def _get_batch_size(self):
        return min(len(self.replay_memory), self.BATCH_SIZE)

    def get_mini_batch(self):

        sample_idx = np.random.choice(len(self.replay_memory), self.BATCH_SIZE)

        batch = np.array(self.replay_memory)[sample_idx]

        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*batch))
        return (states_batch, action_batch, reward_batch, next_states_batch, done_batch)

    def update_replay_memory(self, state, action, reward, next_state, is_done):

        if len(self.replay_memory) > self.MAX_MEMORY:
            self.replay_memory.pop()
        self.replay_memory.append((state, action, reward, next_state, is_done))

    def _run_train_step(self, sess):
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = self.get_mini_batch()

        Q_vals_next = sess.run(self.model.Q_, feed_dict={
            self.model.input_layer: next_states_batch})  # Q_vals_next is shape (BATCH_SIZE, NUM_ACTIONS)

        #print 'Q)Vals'
        #print Q_vals_next


        max_Q_values = np.max(Q_vals_next, axis=1)  # determine the maximum Q Value
        targetQ_score = Q_vals_next

        targetQ_score[np.arange(self.BATCH_SIZE), action_batch] = reward_batch + \
                                                                   ((1.0 - done_batch.astype(int)) * self.gamma * max_Q_values) + \
                                                                   done_batch.astype(int) * -10.0

        loss_update, loss = sess.run([self.model.update, self.model.loss],
                               feed_dict={self.model.input_layer: states_batch,
                                          self.model.Q: targetQ_score})
        return loss, targetQ_score


    def train_model(self):
        with tf.Session() as sess:
            sess.run(self.model.init)
            summary_writer = tf.summary.FileWriter('./tmp/logs', sess.graph)
            merged = tf.summary.merge_all()
            ctr = 0
            for i in range(self.num_episodes):

                obs = self.env.reset()
                is_done = False
                steps = 0
                total_reward = 0.0
                summary = None
                sum_action_value = 0.0
                while True:
                    ctr += 1
                    steps += 1
                    if random.random() < self.RANDOM_ACTION:
                        action = self.env.action_space.sample()
                    else:
                        action, Q_score = sess.run([self.model.predicted_action, self.model.Q_], feed_dict={self.model.input_layer: obs.reshape((1, self.state_space_size))})
                        action = action[0]  # unpack from array
                    sum_action_value += action
                    obs_prime, reward, is_done, info = self.env.step(action)
                    self.update_replay_memory(obs, action, reward, obs_prime, is_done)
                    loss, targetQ_score = self._run_train_step(sess)
                    obs = obs_prime
                    total_reward += reward
                    if is_done:
                        break  # break out of the loop
                summary_writer.add_summary(summary, i)
                self.reward_history.append(total_reward)
                self.step_history.append(steps)
                avg_action_value = sum_action_value / steps
                self.RANDOM_ACTION = max(0.01, self.RANDOM_ACTION * self.RANDOM_ACTION_DECAY)

                print 'episode', i, 'steps', steps, 'eps:', self.RANDOM_ACTION, 'max: ', np.max(targetQ_score), 'len of replay', len(self.replay_memory), 'avg action', avg_action_value, 'SSE', loss
            self.model.saver.save(sess, "./tf_checkpts/model.ckpt")
        summary_writer.close()
        print self.step_history
        print sum(self.step_history)/float(len(self.step_history))

if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    brain = FeedForwardBrain(env)
    brain.train_model()

