#import gym
import tensorflow as tf
import numpy as np
import json



#TODO: add experience replay to the model to train on de-correlated examples
#TODO: add one hidden layer to the network
#TODO: write code to persist the models and recall them. Log test resutsl for determining which architecture are performing well


class TensorFlowModel:
    pass


#need to add exerience replay

class FeedForwardBrain:

    def __init__(self, env, learning_rate = 0.05):
        self.env = env
        self.state_space_size = env.observation_space.n
        self.action_space_size = env.action_space.n
        self.learning_rate = learning_rate
        self.gamma = 0.9
        self.eps = 0.6
        self.num_episodes = 20000
        self.max_steps_per_episode = 100
        self.reward_history = []
        self.step_history = []
        self.model = TensorFlowModel()

    def _build_graph(self):
        self.model.num_epochs = tf.Variable(0)
        self.model.input_layer = tf.placeholder(dtype=tf.float32, shape=(1, self.state_space_size))
        self.model.weights = tf.Variable(tf.random_uniform(shape=(self.state_space_size, self.action_space_size),
                                                minval=0.0,
                                                maxval=0.01))
        self.model.Q_out = tf.matmul(self.model.input_layer, self.model.weights)
        self.model.predicted_action = tf.argmax(self.model.Q_out, 1)
        self.model.nextQ = tf.placeholder(shape=(1, self.action_space_size), dtype=tf.float32)
        self.model.loss = tf.reduce_sum(tf.square(self.model.nextQ - self.model.Q_out))
        self.model.train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.model.update = self.model.train.minimize(self.model.loss)
        tf.summary.scalar('loss', self.model.loss)
        #tf.summary.scalar('train', self.model.train)
        tf.summary.scalar('max_weight', tf.reduce_max(self.model.weights))
        tf.summary.scalar('min_weight', tf.reduce_min(self.model.weights))


        self.model.saver = tf.train.Saver()

        self.model.init = tf.global_variables_initializer()


    def one_hot_encode_state(self,obs):
        """
        observation is an integer representing a tile on the grid
        the encoding is 1 hot encoding where the array is size of state_space
        :param obs:
        :return:
        """
        identity_matrix = np.identity(self.state_space_size)
        return identity_matrix[obs: (obs + 1)]


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

                while steps < self.max_steps_per_episode:
                    ctr += 1
                    steps += 1
                    action, Q_score = sess.run([self.model.predicted_action, self.model.Q_out], feed_dict={self.model.input_layer: self.one_hot_encode_state(obs)})
                    action = action[0]  # unpack from array

                    # with prob eps choose and action from the space at random
                    if np.random.random() < self.eps:
                        action = self.env.action_space.sample()

                    # get new state, reward and other data after taking action
                    obs_prime, reward, is_done, info = self.env.step(action)

                    # given the obs_prime from the prior action determine the Q values for the new state-action pairs
                    # this is doing a SARSA update
                    Q_prime = sess.run(self.model.Q_out, feed_dict={self.model.input_layer: self.one_hot_encode_state(obs_prime)})

                    max_Q_prime = np.max(Q_prime)  # determine the maximum Q Value

                    targetQ_score = Q_score
                    targetQ_score[0, action] = reward + (self.gamma * max_Q_prime)  # update

                    summary, loss_update, weights_update = sess.run([merged, self.model.update, self.model.weights], feed_dict={self.model.input_layer: self.one_hot_encode_state(obs),
                                                                                         self.model.nextQ: targetQ_score
                                                                                         })
                    #print loss_update
                    obs = obs_prime
                    # env.render()
                    total_reward += reward
                    if is_done:
                        #eps = 0.03 * (1.0 - sum(self.reward_history[-100:]) / 100.0)
                        self.eps = (60 - i)/100.0 if i < 60 else 0.01  # reduce the chance of taking a random action only in instances where the model completes
                        break  # break out of the loop

                summary_writer.add_summary(summary, i)
                self.reward_history.append(total_reward)
                self.step_history.append(steps)



                if (i % 500 == 0) and i > 99:
                    print 'episode', i, 'total reward', sum(self.reward_history[-100:]) / 100.0, 'steps', sum(self.step_history[-100:]) / 100.0, 'eps:', self.eps
                    print Q_score
            self.model.saver.save(sess, "./tf_checkpts/model.ckpt")
        summary_writer.close()

if __name__ == "__main__":
    import gym
    env = gym.make('FrozenLake-v0')

    brain = FeedForwardBrain(env)
    brain._build_graph()
    brain.train_model()

