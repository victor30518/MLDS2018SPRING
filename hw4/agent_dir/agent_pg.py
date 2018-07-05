from agent_dir.agent import Agent
import scipy
import numpy as np
import os.path
import tensorflow as tf
from collections import deque
import sys

def prepro(o):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    o = o[35:195]  # crop
    o = o[::2, ::2, 0]  # downsample by factor of 2
    o[o == 144] = 0  # erase background (background type 1)
    o[o == 109] = 0  # erase background (background type 2)
    o[o != 0] = 1  # everything else (paddles, ball) just set to 1
    return o.astype(np.float).ravel()

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.sess = tf.InteractiveSession()

        self.observations = tf.placeholder(tf.float32, [None, 6400])
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(tf.float32, [None, 1], name='advantage')
        self.pi_old = tf.placeholder(tf.float32, [None, 1])
        

        self.h1 = tf.layers.dense(
            self.observations,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        self.h2 = tf.layers.dense(
            self.h1,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.up_probability = tf.layers.dense(
            self.h2,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss1 = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=self.advantage)

        ratio = tf.exp(tf.log((self.up_probability * self.sampled_actions + (1-self.up_probability) * (1-self.sampled_actions))) - tf.log(self.pi_old))
        #ratio = (self.up_probability * self.sampled_actions + (1-self.up_probability) * (1-self.sampled_actions)) / self.pi_old
        self.loss = -tf.reduce_mean(tf.minimum(
            ratio*self.advantage,
            tf.clip_by_value(ratio, 0.8, 1.2)*self.advantage))

        optimizer = tf.train.AdamOptimizer(0.0005)
        self.train_op = optimizer.minimize( 1*self.loss) #+ 0.8*self.loss1 )
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()   

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.saver.restore(self.sess, "./pg_ckpt/policy_network.ckpt")

    def forward_pass(self, observations):
        return self.sess.run(self.up_probability, feed_dict={self.observations: observations.reshape([-1, 6400])})
        

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.init_flag = 1
        self.last_observation = None

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        episode_No = 0
        collect_data = []
        reward_record = []
        reward_list = deque([])
        while True:
            episode_No += 1
            done = False
            reward_sum = 0
            count_w=0
            count_l=0
            
            last_observation = prepro(self.env.reset())
            observation, _, _, _ = self.env.step(self.env.action_space.sample())
            observation = prepro(observation)

            print("Episode: ", episode_No)
            while not done:
                observation_dif = observation - last_observation
                last_observation = observation
                
                if np.random.uniform() < self.forward_pass(observation_dif)[0]: action = 2 #up
                else: action = 3 #down

                observation, reward, done, _ = self.env.step(action)
                observation = prepro(observation)
                reward_sum += reward
                collect_data.append((observation_dif, abs(action-3), reward))

                if reward == 1: count_w+=1
                if reward == -1: count_l+=1

            print("WIN: ", count_w, " lose: ", count_l)
 

            if len(reward_list) < 30:
                reward_list.append(reward_sum)
            else:
                reward_list.append(reward_sum)
                reward_list.popleft()

            print("Total Reward: ", reward_sum, " Avg Reward (last 30 episode): ", np.mean(reward_list))
            reward_record.append(reward_sum)


            states, actions, rewards = zip(*collect_data)
            rewards = self.discount_rewards(rewards)

            states = np.vstack(states)
            actions = np.vstack(actions)
            rewards = np.vstack(rewards)
            
            pi = self.forward_pass(states)
            pi_old = pi * actions + (1-pi) * (1-actions)        
            for i in range(10):
                feed_dict = {self.observations: states, self.sampled_actions: actions, self.advantage: rewards, self.pi_old: pi_old}
                self.sess.run(self.train_op, feed_dict)

            collect_data = []

            self.saver.save(self.sess, 'pg_ckpt/policy_network.ckpt')
            np.save("reward_record.npy",reward_record)
            
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        for i in range(len(rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(i, len(rewards)):
                discounted_reward_sum += rewards[k] * discount
                discount *= 0.99
                if rewards[k] != 0:
                    # Don't count rewards from subsequent rounds
                    break
            discounted_rewards[i] = discounted_reward_sum
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = prepro(observation)
        if self.init_flag == 1:
            self.init_flag = 0
            next_observation, _, _, _ = self.env.step(self.env.get_random_action())
            next_observation = prepro(next_observation)
            observation_dif = next_observation - observation
            self.last_observation = next_observation

        else:
            observation_dif = observation - self.last_observation
            self.last_observation = observation

        if self.forward_pass(observation_dif)[0] > 0.5: #the probability of action_up
            action = 2 #up
        else:
            action = 3 #down
        return action

