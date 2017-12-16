from agent_dir.agent import Agent
from scipy import misc
import numpy as np
from environment import Environment
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

MAX_EPOCH = 500000
MAX_STEPS = 80000



def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = misc.imresize(y, image_size)
    #print('o.shape :', o.shape, '\ty.shape :', y.shape)
    #print(np.expand_dims(resized.astype(np.float32),axis=2).shape)
    return np.expand_dims(resized.astype(np.float32),axis=2)


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
        tf.reset_default_graph()
        self.args = args
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.gamma = 0.99 # reward decay
        self.lr = 0.0001
        self.decay = 0.99
        self.env = env
        self.counter = 0
        #self.batch_size =
        self.resume = False
        self._build_net()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        if args.test_pg or self.resume:
            #you can load your model here           
            saver.restore(self.sess, "Model_pg2/model_pg2.ckpt")
            print('loading trained model')
        else:
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
            self.sess.run(tf.global_variables_initializer())

    def _weight_variables(self, shape, name):
        initializer = tf.random_normal_initializer(mean = 0., stddev = 0.02 )
        return tf.get_variable(shape = shape, initializer = initializer, name = name) 
    def _bias_variables(self, shape, name):
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(shape = shape, initializer = initializer, name = name)
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def _build_net(self):

        # ===================== all inputs =========================
        self.state = tf.placeholder(tf.float32, [None, 80, 80, 1], name = 'state')
        self.reward = tf.placeholder(tf.float32, [None, ], name = 'reward')
        self.action = tf.placeholder(tf.int32 , [None, ], name = 'action')
        # ===================== policy net =========================
        with tf.variable_scope('policy_net'):
            with tf.variable_scope('conv_1'):
                W_conv1 = self._weight_variables([8, 8, 1, 16], name = 'w_conv1')
                b_conv1 = self._bias_variables([16], name = 'b_conv1')
                h_conv1 = tf.nn.relu(self.conv2d(self.state, W_conv1, 4) + b_conv1)
            #    h_conv1 = tf.nn.relu(self.batch_norm((self.conv2d(self.state, W_conv1, 4) + b_conv1), 32, self.is_training, 'CNN'))
            #    h_pool1 = self.max_pool_2x2(h_conv1)
            # 20*20*32
                print('h_conv1.shape : ', h_conv1.shape)
            with tf.variable_scope('conv_2'):
                W_conv2 = self._weight_variables([4, 4, 16, 32], name = 'w_conv2')
                b_conv2 = self._bias_variables([32], name = 'b_conv2')
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
            #    h_conv2 = tf.nn.relu(self.batch_norm((self.conv2d(h_conv1, W_conv2, 2) + b_conv2), 64, self.is_training, 'CNN'))
            #    h_pool2 = self.max_pool_2x2(h_conv2)
            # 10*10*32
                print('h_conv2.shape : ', h_conv2.shape)
                flatten = tf.reshape(h_conv2, [-1, 10*10*32])
            with tf.variable_scope('fc1'):
                W_fc1 = self._weight_variables([10*10*32, 128], name = 'w_fc1')
                b_fc1 = self._bias_variables([128], name = 'b_fc1')

            #    h_fc1 = tf.nn.relu(self.batch_norm((tf.matmul(flatten, W_fc1) + b_fc1), 512, self.is_training, 'FC1'))
                h_fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1)
                #h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            with tf.variable_scope('output'):
                W_fc2 = self._weight_variables([128, 6], name = 'w_fc2')
                b_fc2 = self._bias_variables([6], name = 'b_fc2')
                self.before_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2
                self.policy = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

            with tf.variable_scope('output'):
                #self.log_test = tf.log(self.policy)
                #neg_log_prob = tf.reduce_sum(-tf.log(self.policy)*tf.one_hot(self.action, 6), axis=1)
                neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.before_softmax, labels=self.action)
                self.loss = tf.reduce_sum(neg_log_prob * self.reward)  # reward guided loss

            with tf.name_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr, decay = self.decay).minimize(self.loss)

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.counter = 0

    def test(self, total_episodes=30, seed = 11037):
        rewards = []
        #self.env.seed(seed)
        for i in range(total_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            episode_reward = 0.0

            #playing one game
            while(not done):
                action = self.make_action(state, test=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
        print('Run %d episodes'%(total_episodes))
        print('Mean:', np.mean(rewards))


    def store_ep_memory(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    def _discount_and_norm_rewards(self, r):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            #if r[t] != 0: running_add = 0 
            running_add = running_add * self.gamma + r[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        saver = tf.train.Saver()
        ep_t = 0
        average = []
        for i_episode in range(MAX_EPOCH):
            cur_observation = self.env.reset()
            cur_observation = np.zeros(80*80).reshape((80,80,1))
            init = 0 
            testing = False
            for t in range(MAX_STEPS):
                if i_episode % 400 == 0 and i_episode >= 1000:
                    self.test()
                    testing = True
                    break
                    '''
                    done = False
                    print('\n============================\n')                     
                    for i in range(10):
                        observation = self.env.reset()
                        test_reward = 0
                        done = False
                        init = 0
                        while not done :
                            
                            if init == 0:
                                observation = np.concatenate((prepro(observation), np.zeros(80*80).reshape((80,80,1))), axis=2)
                            else:
                                observation = np.concatenate((observation, diff_observation), axis=2)
                            action = self.make_action(observation, test = True)
                            observation_, reward, done, info = self.env.step(action)
                            diff_observation = prepro(observation_) - observation
                            observation = prepro(observation_)
                            test_reward += reward
                            init += 1
                        print('epoch : ', i, '\ttesting reward: ', test_reward)           
                    break
                    '''      
                else:
                    '''
                    if init == 0:
                        stack_observation = np.concatenate((cur_observation, np.zeros(80*80).reshape((80,80,1))), axis=2)
                    else:
                        diff_observation = cur_observation - pre_observation
                        stack_observation = np.concatenate((cur_observation, diff_observation), axis=2)
                    '''
                    if init != 0:
                        observation = (cur_observation - pre_observation)
                    else:
                        #print(cur_observation)
                        observation = cur_observation
                    action = self.make_action(observation, test = False)
                    #print(action)
                    observation_, reward, done, info = self.env.step(action)
                    #print(observation)
                    #print(reward)
                    self.store_ep_memory(observation, action, reward)
                    #print('shape of observation : ', observation.shape)
                    #diff_observation = prepro(observation_) - observation
                    pre_observation = cur_observation
                    cur_observation = prepro(observation_)
                    #cv2.imshow("image", observation)
                    #cv2.waitKey(1)
                    ep_t += 1
                    init += 1
                if done:
                    break
            if not testing:
                # discount and normalize episode reward
                discounted_ep_rs_norm = self._discount_and_norm_rewards(self.ep_rs)
                #print(np.array(self.ep_obs)[0])
                #print(discounted_ep_rs_norm)
                _, loss, policy = self.sess.run(
                    [self._train_op, self.loss, self.policy],
                    feed_dict={
                        self.state : np.array(self.ep_obs),
                        self.action : np.array(self.ep_as),
                        self.reward : discounted_ep_rs_norm,

                    })
                #self.writer.add_summary(summ, global_step=i_episode)

                save_path = saver.save(self.sess, "Model_pg2/model_pg2.ckpt")
                ep_rs_sum = sum(self.ep_rs)
                if i_episode == 0:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                average.append(ep_rs_sum)
                np.save('pg_avg_reward2.npy', np.array(average))
                print('epoch : ', i_episode, '\tep_rs_sum : ', ep_rs_sum, '\tavg_reward : ', np.mean(average))
                print('ep_t : ', ep_t, "\tEpisode finished after {} timesteps".format(t+1))
                #print(log_test)
                self.ep_obs, self.ep_as, self.ep_rs = [], [], []
            #print(b_fc2)

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
        '''
        if test:
            observation = prepro(observation)
            if self.counter == 0:
                observation_ = np.concatenate((observation, np.zeros(80*80).reshape((80,80,1))), axis=2)
            else:
                diff_observation = observation - self.obs_pre
                observation_ = np.concatenate((observation, diff_observation), axis=2)
            self.obs_pre = observation
            self.counter += 1
        else:
            observation_ = observation
        '''
        if test:
            observation = self.prepro(observation)
            if self.counter == 0:
                observation_ = np.zeros(80*80).reshape((80,80,1))
                #cv2.imshow("image0", observation)
                #cv2.waitKey(1)
            else:
                observation_ = observation - self.obs_pre
            self.counter += 1
            self.obs_pre = observation
            #cv2.imshow("image", observation_)
            #cv2.waitKey(1)
        else:
            observation_ = observation
        norm_observation = (observation_)[np.newaxis, :]
        prob_weights, before_softmax = self.sess.run([self.policy, self.before_softmax], feed_dict={self.state: norm_observation})
        #print('++++++++')
        #print(b_fc2)
        #print('========')
        #print(prob_weights)
        #print(norm_observation)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        #print(action, end='')
        return action

    def prepro(self, o,image_size=[80,80]):
        """
        Call this function to preprocess RGB image to grayscale image if necessary
        This preprocessing code is from
            https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
        
        Input: 
        RGB image: np.array
            RGB screen of game, shape: (210, 160, 3)
        Default return: np.array 
            Grayscale image, shape: (80, 80, 1)
        
        """
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = misc.imresize(y, image_size)
        #print('o.shape :', o.shape, '\ty.shape :', y.shape)
        #print(np.expand_dims(resized.astype(np.float32),axis=2).shape)
        return np.expand_dims(resized.astype(np.float32),axis=2)

