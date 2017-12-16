from agent_dir.agent import Agent
import tensorflow as tf
import gym
from memory_replay import ReplayMemory
import numpy as np

MAX_EPOCH = 500000
MAX_STEPS = 80000


class Agent_Dueling_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_Dueling_DQN,self).__init__(env)

        
        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.gamma = 0.99
        self.lr = 0.0001
        self.epsilon = 1.0 
        self.memory_size = 50000
        self.batch_size = 32
        self.replace_target_iteration = 10000
        self.learn_step_counter = 0
        self._build_net()
        self.memory = ReplayMemory(self.memory_size)
        saver = tf.train.Saver()
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.summaries = tf.summary.merge_all()
        self.sess = tf.Session()
        if args.test_dueling_dqn:
            #you can load your model here           
            saver.restore(self.sess, "Model_deuling/model_deuling_dqn.ckpt")
            print('loading trained model')
        else:
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
            self.sess.run(tf.global_variables_initializer())

    def _weight_variables(self, shape, name):
        initializer = tf.random_normal_initializer(mean = 0., stddev = 0.02)
        return tf.get_variable(shape = shape, initializer = initializer, name = name) 
    def _bias_variables(self, shape, name):
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(shape = shape, initializer = initializer, name = name)
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')   
    def batch_norm(self, x, n_out, is_training, type):
        '''
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        '''
        if type == 'CNN':
            dim_normalization = [0, 1, 2]
        else:
            dim_normalization = [0]
        with tf.variable_scope('bn'):
            beta = tf.get_variable(shape = [n_out], initializer = tf.constant_initializer(0.0), name = 'beta')
            gamma = tf.get_variable(shape = [n_out], initializer = tf.constant_initializer(1.0), name = 'gamma')

            batch_mean, batch_var = tf.nn.moments(x, dim_normalization, name = 'moments')
            ema = tf.train.ExponentialMovingAverage(decay = 0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(is_training, mean_var_with_update,
                                lambda:(ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def _build_net(self):
        # ===================== all inputs =========================
        self.state = tf.placeholder(tf.float32, [None, 84, 84, 4], name = 'state')
        self.state_ = tf.placeholder(tf.float32, [None, 84, 84, 4], name = 'state_')
        self.reward = tf.placeholder(tf.float32, [None, ], name = 'reward')
        self.action = tf.placeholder(tf.int32 , [None, ], name = 'action')
        self.terminal = tf.placeholder(tf.float32, [None, ], name = 'terminal')
        self.is_training = tf.placeholder(tf.bool, name = 'is_training')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        # ===================== build evaluate net =========================
        with tf.variable_scope('eval_net'):
            with tf.variable_scope('conv_1'):
                W_conv1 = self._weight_variables([8, 8, 4, 16], name = 'w_conv1')
                b_conv1 = self._bias_variables([16], name = 'b_conv1')
                h_conv1 = tf.nn.relu(self.conv2d(self.state, W_conv1, 4) + b_conv1)
            #    h_conv1 = tf.nn.relu(self.batch_norm((self.conv2d(self.state, W_conv1, 4) + b_conv1), 16, self.is_training, 'CNN'))
            #    h_pool1 = self.max_pool_2x2(h_conv1)
            # 21*21*32
                print('h_conv1.shape : ', h_conv1.shape)
            with tf.variable_scope('conv_2'):
                W_conv2 = self._weight_variables([4, 4, 16, 32], name = 'w_conv2')
                b_conv2 = self._bias_variables([32], name = 'b_conv2')
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
            #    h_conv2 = tf.nn.relu(self.batch_norm((self.conv2d(h_conv1, W_conv2, 2) + b_conv2), 32, self.is_training, 'CNN'))
            #    h_pool2 = self.max_pool_2x2(h_conv2)
            # 11*11*64
                print('h_conv2.shape : ', h_conv2.shape)
            with tf.variable_scope('conv_3'):
                W_conv3 = self._weight_variables([3, 3, 32, 64], name = 'w_conv3')
                b_conv3 = self._bias_variables([64], name = 'b_conv3')
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
            #    h_conv3 = tf.nn.relu(self.batch_norm((self.conv2d(h_conv2, W_conv3, 1) + b_conv3), 64, self.is_training, 'CNN'))
            #    h_pool3 = self.max_pool_2x2(h_conv3)
                print('h_conv3.shape : ', h_conv3.shape)
                flatten = tf.reshape(h_conv3, [-1, 11*11*64])
            # 11*11*64
            with tf.variable_scope('fc1'):
                W_adv1 = self._weight_variables([11*11*64, 512], name = 'w_adv1')
                b_adv1 = self._bias_variables([512], name = 'b_adv1')
                h_adv1 = tf.nn.relu(tf.matmul(flatten, W_adv1) + b_adv1)

                W_val1 = self._weight_variables([11*11*64, 512], name = 'w_val1')
                b_val1 = self._bias_variables([512], name = 'b_val1')
                h_val1 = tf.nn.relu(tf.matmul(flatten, W_val1) + b_val1)

            with tf.variable_scope('output'):
                W_adv2 = self._weight_variables([512, 4], name = 'w_adv2')
                b_adv2 = self._bias_variables([4], name = 'b_adv2')
                eval_advantage = tf.matmul(h_adv1, W_adv2) + b_adv2

                W_val2 = self._weight_variables([512, 1], name = 'w_val2')
                b_val2 = self._bias_variables([1], name = 'b_val2')
                eval_value = tf.matmul(h_val1, W_val2) + b_val2
                # Average Deuling
                self.q_eval = eval_value + (eval_advantage - tf.reduce_mean(eval_advantage, axis=1, keep_dims = True))
        # ===================== build target net =========================
        with tf.variable_scope('target_net'):
            with tf.variable_scope('conv_1'):
                W_conv1 = self._weight_variables([8, 8, 4, 16], name = 'w_conv1')
                b_conv1 = self._bias_variables([16], name = 'b_conv1')
                h_conv1 = tf.nn.relu(self.conv2d(self.state_, W_conv1, 4) + b_conv1)
            #    h_conv1 = tf.nn.relu(self.batch_norm((self.conv2d(self.state_, W_conv1, 4) + b_conv1), 16, self.is_training, 'CNN'))
            #    h_pool1 = self.max_pool_2x2(h_conv1)
            # 21*21*32
                print('h_conv1.shape : ', h_conv1.shape)
            with tf.variable_scope('conv_2'):
                W_conv2 = self._weight_variables([4, 4, 16, 32], name = 'w_conv2')
                b_conv2 = self._bias_variables([32], name = 'b_conv2')
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
            #    h_conv2 = tf.nn.relu(self.batch_norm((self.conv2d(h_conv1, W_conv2, 2) + b_conv2), 32, self.is_training, 'CNN'))
            #    h_pool2 = self.max_pool_2x2(h_conv2)
            # 11*11*64
                print('h_conv2.shape : ', h_conv2.shape)
            with tf.variable_scope('conv_3'):
                W_conv3 = self._weight_variables([3, 3, 32, 64], name = 'w_conv3')
                b_conv3 = self._bias_variables([64], name = 'b_conv3')
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
            #    h_conv3 = tf.nn.relu(self.batch_norm((self.conv2d(h_conv2, W_conv3, 1) + b_conv3), 64, self.is_training, 'CNN'))
            #    h_pool3 = self.max_pool_2x2(h_conv3)
                print('h_conv3.shape : ', h_conv3.shape)
                flatten = tf.reshape(h_conv3, [-1, 11*11*64])
            # 11*11*64
            with tf.variable_scope('fc1'):
                W_adv1 = self._weight_variables([11*11*64, 512], name = 'w_adv1')
                b_adv1 = self._bias_variables([512], name = 'b_adv1')
                h_adv1 = tf.nn.relu(tf.matmul(flatten, W_adv1) + b_adv1)

                W_val1 = self._weight_variables([11*11*64, 512], name = 'w_val1')
                b_val1 = self._bias_variables([512], name = 'b_val1')
                h_val1 = tf.nn.relu(tf.matmul(flatten, W_val1) + b_val1)

            with tf.variable_scope('output'):
                W_adv2 = self._weight_variables([512, 4], name = 'w_adv2')
                b_adv2 = self._bias_variables([4], name = 'b_adv2')
                target_advantage = tf.matmul(h_adv1, W_adv2) + b_adv2

                W_val2 = self._weight_variables([512, 1], name = 'w_val2')
                b_val2 = self._bias_variables([1], name = 'b_val2')
                target_value = tf.matmul(h_val1, W_val2) + b_val2        
                self.q_next = target_value + (target_advantage - tf.reduce_mean(target_advantage, axis=1, keep_dims = True))

        with tf.variable_scope('q_target'):
            #q_target = tf.cond(self.terminal, lambda : self.reward + self.gamma * tf.reduce_max(self.q_next, axis = 1, name = 'Qmax_s_'), lambda: self.reward)
            q_target = self.reward + self.gamma * (1 - self.terminal) * tf.reduce_max(self.q_next, axis = 1, name = 'Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype = tf.int32), self.action], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
        with tf.variable_scope('loss'):
            #self.delta = (self.q_target - self.q_eval_wrt_a)
            #clipped_error = tf.where(tf.abs(self.delta) < 1.0,
            #                        0.5 * tf.square(self.delta),
            #                        tf.abs(self.delta) - 0.5, name='clipped_error')
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            #self.loss = tf.reduce_mean(clipped_error, name = 'loss')
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)



    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass
    def test(self, total_episodes=100, seed = 11037):
        rewards = []
        self.env.seed(seed)
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

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        saver = tf.train.Saver()
        rewards = []
        ep_t = 0
        for i_episode in range(MAX_EPOCH):
            observation = self.env.reset()
            ep_reward = 0 
            for t in range(MAX_STEPS):
                if i_episode % 500 == 0 and i_episode != 0:
                    print('\n============================\n')
                    self.test()
                    print('\n============================\n')
                    break
                else:
                    action = self.make_action(observation, test = False)
                    observation_, reward, done, info = self.env.step(action)
                    self.memory.store(observation, action, reward, done)
           
                    #print('shape of observation : ', observation.shape)
                    observation = observation_
                    ep_reward += reward
                    if self.epsilon > 0.05:
                        self.epsilon -= (1 - 0.05)/1000000
                    else:
                        self.epsilon = 0.05

                    ep_t += 1

                    if ep_t % self.replace_target_iteration == 0:
                        self.sess.run(self.target_replace_op)
                        print('\n target parameter replaced : ', str(ep_t / self.replace_target_iteration), '\n')
                        save_path = saver.save(self.sess, "Model_deuling_2/model_deuling_dqn_2.ckpt")
                if ep_t % 4 == 0 and ep_t > self.batch_size:
                    if self.memory.current >= self.memory_size:
                        s, a, r, ter, s_ = self.memory.sample_memory(np.random.choice(self.memory_size - 1, size=self.batch_size))
                    else:
                        s, a, r, ter, s_ = self.memory.sample_memory(np.random.choice(self.memory.current - 1, size=self.batch_size))

                    _, loss, summ, q_value = self.sess.run(
                        [self._train_op, self.loss, self.summaries, self.q_eval],
                        feed_dict={
                            self.state : s,
                            self.action : a,
                            self.reward : r,
                            self.state_ : s_,
                            self.terminal : ter,
                            self.is_training : True,
                            self.keep_prob : 1.0,
                        })
                    self.writer.add_summary(summ, global_step=i_episode)
                    self.learn_step_counter += 1
                    #print('learn_step_counter : ', self.learn_step_counter, '\t loss : ', loss)
                    #print('q_value : ', q_value[0])    
                if done:
                    break
                #total_reward = tf.constant(ep_reward, name = 'r')
                #tf.summary.scalar('reward', total_reward)
                #print('epoch : ', i_episode)
                #summ = self.sess.run(self.summaries)
                #self.writer.add_summary(summ, global_step=i_episode)
            rewards.append(ep_reward)
            np.save('dueling_dqn_reward.npy', np.array(rewards))
            if ep_t > self.batch_size :
                print('q_value : ', q_value[0])    
            print('epoch : ', i_episode, '\ttotal_reward : ', ep_reward, '\taverage_reward : ', np.mean(rewards), '\teplison : ', self.epsilon)
            print('ep_t : ', ep_t, "\tEpisode finished after {} timesteps".format(t+1))
                           
            
                
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        norm_observation = (observation)[np.newaxis, :]
        if not test:
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.env.get_random_action()
            else:
                action_value = self.sess.run(self.q_eval, feed_dict={self.state: norm_observation, self.is_training : True, self.keep_prob : 1.0})
                action = np.argmax(action_value)
        else:
            if np.random.uniform(0, 1) < 0.01:
                action = self.env.get_random_action()
            else:
                action_value = self.sess.run(self.q_eval, feed_dict={self.state: norm_observation, self.is_training : False, self.keep_prob : 1.0})
                action = np.argmax(action_value)
        return action

