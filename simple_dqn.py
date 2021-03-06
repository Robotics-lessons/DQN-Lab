#!/usr/bin/python3

import gym
import random

import numpy as np
import tensorflow as tf
from gym import wrappers, logger
import matplotlib.pyplot as plt
import torch
import argparse
import pickle

class DQN:
  REPLAY_MEMORY_SIZE = 10000
  RANDOM_ACTION_PROB = 0.5
  RANDOM_ACTION_DECAY = 0.99
  HIDDEN1_SIZE = 128
  HIDDEN2_SIZE = 128
  NUM_EPISODES = 5000
  MAX_STEPS = 1000
  LEARNING_RATE = 0.0001
  MINIBATCH_SIZE = 10
  DISCOUNT_FACTOR = 0.9
  TARGET_UPDATE_FREQ = 100
  REG_FACTOR = 0.001
  LOG_DIR = '/tmp/dqn'
  episode_durations = []
  SOLVED_T = 199
  STREAK_TO_END = 100
  TORCH_MODEL = 'simple_dqn.ckpt'

  def __init__(self, env):
    self.env = gym.make(env)
    assert len(self.env.observation_space.shape) == 1
    self.input_size = self.env.observation_space.shape[0]
    self.output_size = self.env.action_space.n
    self.Q = np.zeros([self.input_size, 2])
    
  def init_network(self):
    # Inference
    self.x = tf.placeholder(tf.float32, [None, self.input_size])
    with tf.name_scope('hidden1'):
      W1 = tf.Variable(
                 tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], 
                 stddev=0.01), name='W1')
      b1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name='b1')
      h1 = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
    with tf.name_scope('hidden2'):
      W2 = tf.Variable(
                 tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], 
                 stddev=0.01), name='W2')
      b2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name='b2')
      h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
    with tf.name_scope('output'):
      W3 = tf.Variable(
                 tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size], 
                 stddev=0.01), name='W3')
      b3 = tf.Variable(tf.zeros(self.output_size), name='b3')
      self.Q = tf.matmul(h2, W3) + b3
    self.weights = [W1, b1, W2, b2, W3, b3]

    # Loss
    self.targetQ = tf.placeholder(tf.float32, [None])
    self.targetActionMask = tf.placeholder(tf.float32, [None, self.output_size])
    # TODO: Optimize this
    q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask), 
                  reduction_indices=[1])
    self.loss = tf.reduce_mean(tf.square(tf.subtract(q_values, self.targetQ)))

    # Reguralization
    for w in [W1, W2, W3]:
      self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))

    # Training
    optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = optimizer.minimize(self.loss, global_step=global_step)

  def train(self, num_episodes=NUM_EPISODES):
    replay_memory = []
    num_streaks = 0

    self.session = tf.Session()

    # Summary for TensorBoard
    tf.summary.scalar('loss', self.loss)
    self.summary = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.LOG_DIR, self.session.graph)

    self.session.run(tf.initialize_all_variables())
    total_steps = 0

    for episode in range(num_episodes):
      print("Training: Episode = %d, Global step = %d" % (episode, total_steps), end='')
      state = self.env.reset()
      target_weights = self.session.run(self.weights)

      for step in range(self.MAX_STEPS):
        # Pick the next action and execute it
        action = None
        if random.random() < self.RANDOM_ACTION_PROB:
          action = self.env.action_space.sample()
        else:
          q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
          action = q_values.argmax()
        self.RANDOM_ACTION_PROB *= self.RANDOM_ACTION_DECAY
        obs, reward, done, _ = self.env.step(action)

        # Update replay memory
        if done:
          reward = -100
        replay_memory.append((state, action, reward, obs, done))
        if len(replay_memory) > self.REPLAY_MEMORY_SIZE:
          replay_memory.pop(0)
        state = obs

        # Sample a random minibatch and fetch max Q at s'
        if len(replay_memory) >= self.MINIBATCH_SIZE:
          minibatch = random.sample(replay_memory, self.MINIBATCH_SIZE)
          next_states = [m[3] for m in minibatch]
          # TODO: Optimize to skip terminal states
          feed_dict = {self.x: next_states}
          feed_dict.update(zip(self.weights, target_weights))
          q_values = self.session.run(self.Q, feed_dict=feed_dict)
          max_q_values = q_values.max(axis=1)

          # Compute target Q values
          target_q = np.zeros(self.MINIBATCH_SIZE)
          target_action_mask = np.zeros((self.MINIBATCH_SIZE, self.output_size), dtype=int)
          for i in range(self.MINIBATCH_SIZE):
            _, action, reward, _, terminal = minibatch[i]
            target_q[i] = reward
            if not terminal:
              target_q[i] += self.DISCOUNT_FACTOR * max_q_values[i]
            target_action_mask[i][action] = 1

          # Gradient descent
          states = [m[0] for m in minibatch]
          feed_dict = {
            self.x: states, 
            self.targetQ: target_q,
            self.targetActionMask: target_action_mask,
          }
          _, summary = self.session.run([self.train_op, self.summary], 
                                        feed_dict=feed_dict)

          # Write summary for TensorBoard
          if total_steps % 100 == 0:
            self.summary_writer.add_summary(summary, total_steps)

          # Update target weights
          if total_steps % self.TARGET_UPDATE_FREQ == 0:
            target_weights = self.session.run(self.weights)

        total_steps += 1
        if done:
          if (step >= self.SOLVED_T):
            num_streaks += 1
          else:
            num_streaks = 0
          break
      print(", duration = %d" % step)    
      self.episode_durations.append(step + 1)
      self.plot_durations()
      if num_streaks > 2: # self.STREAK_TO_END:
        checkpoint = {'Q': self.Q}
        with open(self.TORCH_MODEL, 'wb') as f:
            torch.save(checkpoint, f)
        break

  def play(self, model_file_name = ''):
    state = self.env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
      self.env.render()
      if len(model_file_name) > 0:
        print('Load model file = ', model_file_name)
#        self.Q = torch.load(self.TORCH_MODEL)
        with open(model_file_name, 'rb') as f:
          checkpoint = torch.load(f)
        self.Q = checkpoint['Q']
        self.Q.eval()
       
      q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
      action = q_values.argmax()
      state, _, done, _ = self.env.step(action)
      steps += 1
    return steps

  def close(self):
    self.env.close()

  def plot_durations(self):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(self.episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
#    plt.plot(durations_t.numpy())
    plt.plot(self.episode_durations)
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color='r')


    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-model', action="store", dest='model_file', default='')
  results = parser.parse_args()
  print('input model file = ', results.model_file)
 # exit()
  dqn = DQN('CartPole-v0')
  dqn.init_network()

#  dqn.env.monitor.start('/tmp/cartpole')
#  dqn.env = wrappers.Monitor(dqn.env, directory='/tmp/cartpole', force=True)
  if len(results.model_file) == 0:
    dqn.train()
 # dqn.env.monitor.close()

  res = []
  for i in range(50):
    steps = dqn.play('simple_dqn.ckpt')
    print("Testing: Episode = %d, Test steps = %d" % (i, steps))
    res.append(steps)
  print("Mean steps = ", sum(res) / len(res))
  dqn.close()

