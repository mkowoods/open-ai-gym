import gym
import tensorflow as tf
import numpy as np


print "action_space", env.action_space
print "observation_space", env.observation_space

state_space_size = env.observation_space.n
action_space_size = env.action_space.n
learning_rate = 0.1
# defines shape for input layer
input_layer = tf.placeholder(dtype=tf.float32, shape=(1, state_space_size))
weights = tf.Variable(tf.random_uniform(shape=(state_space_size, action_space_size),
                                        minval=0.0,
                                        maxval=0.01))
Q_out = tf.matmul(input_layer, weights)
predicted_action = tf.argmax(Q_out, 1)
nextQ = tf.placeholder(shape=(1, action_space_size), dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Q_out))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
update = train.minimize(loss)

saver = tf.train.Saver()


#
def encode_state(obs):
    """
    observation is an integer representing a tile on the grid
    the encoding is 1 hot encoding where the array is size of state_space
    :param obs:
    :return:
    """
    identity_matrix = np.identity(state_space_size)
    return identity_matrix[obs: (obs + 1)]


# training
init = tf.initialize_all_variables()
# inputs represented as


gamma = 0.9
eps = 0.1
num_episodes = 4000
max_steps = 100

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./tf_checkpts/model.ckpt")
    reward_history = []
    for i in range(num_episodes):

        obs = env.reset()
        is_done = False
        steps = 0
        total_reward = 0.0

        while steps < max_steps:
            steps += 1
            action, Q_score = sess.run([predicted_action, Q_out], feed_dict={input_layer: encode_state(obs)})
            action = action[0]  # unpack from array
            # print 'q_score', Q_score
            # print 'action', action

            # with prob eps choose and action from the space at random
            if np.random.random() < eps:
                action = env.action_space.sample()

            # get new state, reward and other data after taking action
            obs_prime, reward, is_done, info = env.step(action)

            # given the obs_prime from the prior action determine the Q values for the new state-action pairs
            # this is doing a SARSA update
            Q_prime = sess.run(Q_out, feed_dict={input_layer: encode_state(obs_prime)})

            max_Q_prime = np.max(Q_prime)  # determine the maximum Q Value

            targetQ_score = Q_score
            targetQ_score[0, action] = reward + (gamma * max_Q_prime)  # update

            loss_update, weights_update = sess.run([update, weights], feed_dict={input_layer: encode_state(obs),
                                                                                 nextQ: targetQ_score
                                                                                 })
            # print train_results
            # print weights
            obs = obs_prime
            # env.render()
            total_reward += reward
            if is_done:
                eps = 1.0 / ((
                             i / 50.0) + 10)  # reduce the chance of taking a random action only in instances where the model completes
                break  # break out of the loop
        reward_history.append(total_reward)
        if (i % 100 == 0) and i > 99:
            print 'episode', i, 'total reward', sum(reward_history[-100:]) / 100.0, 'steps', steps
    saver.save(sess, "./tf_checkpts/model.ckpt")

    env.monitor.close()
