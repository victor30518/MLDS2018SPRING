"""
reference:
https://github.com/floodsung/DQN-Atari-Tensorflow/blob/master/BrainDQN_Nature.py
"""
import numpy as np 
import random
np.random.seed(631)
random.seed(631)
from agent_dir.agent import Agent
import tensorflow as tf 
from collections import deque
import os

UPDATE_Qhat_TIME = 10000
UPDATE_Q_TIME = 4
GAMMA = 0.95 
END_EPSILON = 0.1
INIT_EPSILON = 1.0
REPLAY_BUFFER = 10000 
MODEL_NAME = "./Best"
BATCH_SIZE = 32

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        
        self.env = env
        self.args = args
        self.epsilon = INIT_EPSILON
        self.actions = env.action_space.n
        # init
        self.timeStep = 0
        self.replayBuffer = deque()
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
        # assign value(copy net)
        self.TargetQupdate = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

        # define loss function & placeholder
        self.actionInput = tf.placeholder("float",[None,self.actions])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.loss = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.loss)        
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))
        self.session.run(tf.global_variables_initializer())

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.saver.restore(self.session, MODEL_NAME)
            print("Compute the score......")

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4,4,32,64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([3136,512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512,self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer
        stateInput = tf.placeholder("float",[None,84,84,4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
        h_conv3_shape = h_conv3.get_shape().as_list()
        h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    def copyTargetQNetwork(self):
        self.session.run(self.TargetQupdate)

    def trainQNetwork(self):

        
        # random sample minibatch from replay memory
        minibatch = random.sample(self.replayBuffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # compute y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={self.yInput : y_batch, self.actionInput : action_batch, self.stateInput : state_batch})

        # save network
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, './save_model/Model', global_step = self.timeStep)
        
        # update target Q
        if self.timeStep % UPDATE_Qhat_TIME == 0:
            self.copyTargetQNetwork()

    def make_action(self,observation, test=True):
        QValue = self.QValue.eval(feed_dict={self.stateInput:observation.reshape((1,84,84,4))})[0]
        
        if random.random() <= self.epsilon and not test:
            action = random.randrange(self.actions)
        elif test and random.random()>0.01:
        	action = np.argmax(QValue)
        elif test:
            action = random.randrange(self.actions)        
        else:
            action = np.argmax(QValue)

        if self.epsilon > END_EPSILON and self.timeStep > 50000.: 
            self.epsilon -= (INIT_EPSILON - END_EPSILON) / 1000000.
            
        return action


    def setPerception(self,current_state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.actions)
        one_hot_action[action] = 1
        self.replayBuffer.append((current_state,one_hot_action,reward,next_state,done))
        if len(self.replayBuffer) > REPLAY_BUFFER:
            self.replayBuffer.popleft()
        if len(self.replayBuffer) > BATCH_SIZE:
            if self.timeStep % UPDATE_Q_TIME ==0:
                self.trainQNetwork()
        self.timeStep += 1

    def train(self):
        """
        Implement your training algorithm here
        """
        records = [] #save to plot learning curve
        for e in range(1000000):
            current_state = self.env.reset()
            step_count = 0
            total_reward = 0

            for ct in range(10000):
                next_state,reward, done, _ = self.env.step(self.make_action(current_state, test=False))
                # self clip
                unclip_reward = reward
                total_reward += unclip_reward
                if reward>0:
                    reward = np.clip(reward, a_min=None, a_max=1.) #self clip
                
                self.setPerception(current_state,action,reward,next_state, done)
                current_state = next_state

                step_count +=1 

                if done == True:
                    print("episode:", e, " reward:",total_reward)
                    records.append((e,step_count,total_reward,self.timeStep))
                    break

            np.save("learning_curve_unclip.npy", np.array(records))

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
