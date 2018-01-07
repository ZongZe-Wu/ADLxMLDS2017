import tensorflow as tf
from DCGAN import *
import argparse,sys
import time
import matplotlib.pyplot as plt
from scipy.misc import imsave
import re
from preprocess import Sent2Vec
import collections
import random
class D_batch_gen(object):
	def __init__(self, right_training_image, right_training_text, str_tag):
		self.right_training_image = right_training_image
		self.right_training_text = right_training_text
		#self.wrong_training_image = wrong_training_image
		#self.wrong_training_text = wrong_training_text
		self.str_tag = str_tag
		self.amount = int(self.right_training_text.shape[0])
		self._index = 0

		self.color_hair = ['orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair',
		'blue hair', 'black hair', 'brown hair', 'blonde hair', 'unknown hair']
		self.color_eyes = ['gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
		'green eyes', 'brown eyes', 'red eyes', 'blue eyes', 'unknown eyes']
		self.len_hair = len(self.color_hair)
		self.len_eyes = len(self.color_eyes)
		feature_dict = np.load('feature_dict.npy', encoding='bytes').item()
		self.new_feature_dict = {}
		for key, value in feature_dict.items():
			key = key.decode("utf-8")
			self.new_feature_dict[key] = value
		self.num_label = np.load('training/num_label.npy')
		self.image_order = np.load('training/image_order.npy').tolist()
		inv_map = np.load('training/inv_map.npy')
		self.inv_map = inv_map.item()
		#self.od_preprocess_features = np.load('training/od_preprocess_features.npy')
	def __call__(self, batch_size):
		self.index = self._index%self.amount
		right_image = self.right_training_image[self.index:self.index+batch_size]
		right_text = self.right_training_text[self.index:self.index+batch_size]
		num_label = self.num_label[self.index:self.index+batch_size]
		wrong_text = self.wrong_tags(num_label)
		od_preprocess_features = self.str_tag[self.index:self.index+batch_size]
		wrong_image = self.wrong_image(od_preprocess_features)
		#wrong_image = self.wrong_training_image[self.index:self.index+batch_size]
		#wrong_text = self.wrong_training_text[self.index:self.index+batch_size,1:]
		self._index += batch_size 
		return right_image, right_text, wrong_image, wrong_text
	def choose_tags(self, hair, eyes):
		buf = [0]*13
		buf1 = [0]*12
		buf[hair]=1
		buf1[eyes]=1
		return buf+buf1
	def wrong_tags(self, tags):
		#wrong_tags_vec = [self.new_feature_dict[self.color_hair[random.choice(list(range(0,value[0]))+list(range(value[0]+1,self.len_hair)))]+' '+ \
		#				self.color_eyes[random.choice(list(range(0,value[1]))+list(range(value[1]+1,self.len_eyes)))]] for value in tags]
		wrong_tags_vec = [self.choose_tags(random.choice(list(range(0,value[0]))+list(range(value[0]+1,self.len_hair))), random.choice(list(range(0,value[1]))+list(range(value[1]+1,self.len_eyes)))) for value in tags]

		#print(np.array(wrong_tags_vec).shape)
		return wrong_tags_vec

	def wrong_image(self, od_preprocess_features):
		wrong_od_image = []
		for value in od_preprocess_features:
			while(True):
				inv_key = random.choice(list(self.inv_map.keys()))
				#print(value, ' ', inv_key)
				a = re.split(' ',inv_key)
				b = re.split(' ', value)
				if a[0] != b[0] and a[2] != b[2]:
					#print(value, ' ', inv_key)
					break
			wrong_od_image.append(self.right_training_image[self.image_order.index(random.choice(self.inv_map[inv_key]))])
		return wrong_od_image
class G_batch_gen(object):
	def __init__(self, right_training_image, right_training_text, str_tag):
		self.right_training_image = right_training_image
		self.right_training_text = right_training_text
		#self.wrong_training_image = wrong_training_image
		#self.wrong_training_text = wrong_training_text
		self.str_tag = str_tag
		self.amount = int(self.right_training_text.shape[0])
		self._index = 0
	def __call__(self, batch_size):
		self.index = self._index%self.amount
		right_image = self.right_training_image[self.index:self.index+batch_size]
		right_text = self.right_training_text[self.index:self.index+batch_size]
		#wrong_image = self.wrong_training_image[self.index:self.index+batch_size]
		#wrong_text = self.wrong_training_text[self.index:self.index+batch_size,1:]
		tag = self.str_tag[self.index:self.index+batch_size]
		self._index += batch_size 
		return right_image, right_text, tag

# def train(self):
# 	self.sess.run(tf.global_variables_initializer())
# 	epsilon = tf.random_uniform([], 0.0, 1.0)
# 	for right_image, right_text, wrong_image, wrong_text in batch_gen(self.batch_size):
# 		# (real image, real text) (fake image, right text)
# 		# (real image, wrong text) (wrong image, right text)
# 		self.x_ = self.sess.run(self.Gz, feed_dict={self.z : self.z_sampler(self.batch_size, self.z_dim), self.text : right_text})
# 		self.d = self.sess.run(self.Dx, feed_dict={self.x : right_image, self.text : right_text})
# 		self.d_ = self.sess.run(self.Dg, feed_dict={self.x : self.x_, self.text : right_text})
# 		self.d_real_wrong = self.sess.run(self.Dx, feed_dict={self.x : right_image, self.text : wrong_text})
# 		self.d_wrong_real = self.sess.run(self.Dx, feed_dict={self.x : wrong_image, self.text : right_text})

# 		self.g_loss = -tf.reduce_mean(self.d_)
# 		self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_) - tf.reduce_mean(self.d_real_wrong) - tf.reduce_mean(self.d_wrong_real)
# 		x_hat = epsilon * self.x + (1 - epsilon) * self.x_
def z_sampler(batch_size, z_dim):
	#return np.random.normal(0.0, 1.0, size=[batch_size, z_dim])
	return np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim])
class WDCGAN(object):
	def __init__(self, argv, G_Net, D_Net, Embedding, G_bg=None, D_bg=None):
		self.argv = argv
		#self.args = args
		self.z_dim = 200
		self.g_net = G_Net
		self.d_net = D_Net
		self.emb_net = Embedding
		self.d_batch_gen = D_bg
		self.g_batch_gen = G_bg
		self.batch_size = 64
		self.color_hair = ['orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair',
		'blue hair', 'black hair', 'brown hair', 'blonde hair', 'unknown hair']
		self.color_eyes = ['gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
		'green eyes', 'brown eyes', 'red eyes', 'blue eyes', 'unknown eyes']
		self.x = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')	
		#self.right_image = tf.placeholder(tf.float32, [None, 64, 64, 3], name='right_image')
		self.wrong_image = tf. placeholder(tf.float32, [None, 64, 64, 3], name='wrong_image')
		self.right_text = tf.placeholder(tf.float32, [None, 25], name='right_text')
		self.wrong_text = tf.placeholder(tf.float32, [None, 25], name='wrong_text')

		#self.right_enc_text = self.emb_net(self.right_text, reuse_variables=False)
		#self.wrong_enc_text = self.emb_net(self.wrong_text)

		# (fake image)
		self.x_ = self.g_net(self.z, self.right_text)

		# discriminator
		# (real image, real text)
		self.d = self.d_net(self.x, self.right_text, reuse_variables=False)
		# (fake image, right text))
		self.d_ = self.d_net(self.x_, self.right_text)
		# (real image, wrong text)
		self.r_w_d = self.d_net(self.x, self.wrong_text)
		# (wrong image, right text)
		self.w_r_d = self.d_net(self.wrong_image, self.right_text)

		#self.f_w_d = self.d_net(self.x_, self.wrong_enc_text)
		#self.g_loss = -tf.reduce_mean(self.d_)
		#self.g_loss = tf.reduce_mean(self.d_)
		#self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)/3 - tf.reduce_mean(self.r_w_d)/3 - tf.reduce_mean(self.w_r_d)/3
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d, labels = tf.ones_like(self.d)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_, labels = tf.zeros_like(self.d_)))
		self.d_loss_r_w = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.r_w_d, labels = tf.zeros_like(self.r_w_d)))
		self.d_loss_w_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.w_r_d, labels = tf.zeros_like(self.w_r_d)))
		self.d_loss = self.d_loss_real + (self.d_loss_fake+self.d_loss_r_w+self.d_loss_w_r)/3
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_, labels = tf.ones_like(self.d_)))
		'''
		epsilon = tf.random_uniform([], 0.0, 1.0)
		x_hat = epsilon * self.x + (1 - epsilon) * self.x_

		d_hat = self.d_net(x_hat, self.right_enc_text)

		ddx = tf.gradients(d_hat, x_hat)[0]
		#print(ddx.get_shape().as_list())
		ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
		scale = 10.0
		self.ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
		self.d_loss = self.d_loss + self.ddx
		'''
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.d_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.6, beta2=0.9).minimize(self.d_loss, var_list=self.d_net.vars)
			#self.g_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.6, beta2=0.9).minimize(self.g_loss, var_list=[self.g_net.vars, self.emb_net.vars])
			self.g_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.6, beta2=0.9).minimize(self.g_loss, var_list=self.g_net.vars)
		self.sess = tf.Session()
		writer = tf.summary.FileWriter("logs/", self.sess.graph)
		'''
		feature_dict = np.load('feature_dict.npy', encoding='bytes').item()
		new_feature_dict = {}
		for key, value in feature_dict.items():
			key = key.decode("utf-8")
			new_feature_dict[key] = value
		self.inv_map = {}
		for k, v in new_feature_dict.items():
			self.inv_map[v] = k
		'''
		'''
		self.testing_tag=[]
		self.testing_name=[]
		with open('data/sample_testing_text.txt') as testing_file:
			for line in testing_file:
				input_buffer = [i.strip() for i in re.split(',',line.strip())]
				buf = input_buffer[1].split(' ')
				self.testing_name.append(input_buffer[1])
				self.testing_tag.append(self.choose_tags(self.color_hair.index(buf[0]+' '+buf[1]), self.color_eyes.index(buf[2]+' '+buf[3])))
		'''
		#vec_features = Sent2Vec(self.tag_test)
		#self.vec_features = np.array(vec_features)[:,1:]
		#if self.args.train:
		#	self.train()
		#if self.args.test:
		self.test()
	def choose_tags(self, hair, eyes):
		buf = [0]*13
		buf1 = [0]*12
		buf[hair]=1
		buf1[eyes]=1
		return buf+buf1
	def train(self, resume=False,batch_size=64, batch_epoch=500000):
		if resume:
			saver = tf.train.Saver()
			saver.restore(self.sess, "Model6/model.ckpt")
			print('loading trained model')
		else:
			self.sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
		#plt.ion()
		d_iters = 5
		g_iters = 1
		for ep in range(batch_epoch):
			timestamp1 = time.time()
			for i in range(0, d_iters):
				r_i, r_t, w_i, w_t = self.d_batch_gen(batch_size)
				self.batch_size = r_t.shape[0]
				bz = z_sampler(self.batch_size, self.z_dim)
				feed_dict = {self.x : r_i, self.z : bz,\
				 			self.right_text : r_t, self.wrong_text : w_t,\
				 			self.wrong_image : w_i }
				_, dis_loss=self.sess.run([self.d_train_op, self.d_loss], feed_dict=feed_dict)
				#print('penalty', penalty)
			for j in range(0, g_iters):
				r_i, r_t, tag = self.g_batch_gen(batch_size)
				self.batch_size = r_t.shape[0]
				#print(self.batch_size)
				bz = z_sampler(self.batch_size, self.z_dim)
				feed_dict={self.z : bz, self.right_text : r_t}
				_, gen_loss, g_image = self.sess.run([self.g_train_op, self.g_loss, self.x_], feed_dict=feed_dict)
			timestamp2 = time.time()
			#print("%.2f seconds for reading text and transfer into skip-thought vector" % (timestamp2 - timestamp1))
			print('Epoch : ', ep, '\ttime %.2f: ' % (timestamp2 - timestamp1), '\tD_LOSS : ', dis_loss, '\tG_LOSS : ', gen_loss)
			#if (ep+1) % 500 == 0:
								
				#print(self.inv_map[r_t[test_col,:]])
				#for test_col in range(self.batch_size): 
				#self.test()
			if (ep+1)%50 == 0:
				#test_col = random.randint(0,self.batch_size-1)
				#fig = plt.figure()
				#plt.imshow(g_image[0])
				#fig.savefig('image/'+'plot_'+'ep-'+str(ep+1)+'.png')
				#fig.canvas.set_window_title(r_t[0])
				
				test_col = random.randint(0,self.batch_size-1)
				imsave('image_8/'+'ep-'+str(ep+1)+'-'+tag[test_col]+'.jpg', g_image[test_col])
				bz = z_sampler(int(len(self.testing_tag)), self.z_dim)
				feed_dict={self.z : bz, self.right_text : self.testing_tag}
				g_image = self.sess.run(self.x_, feed_dict=feed_dict)
				for count in range(int(len(self.testing_tag))):
					imsave('image_testing6/'+'sample_'+str(count+1)+'_'+str(self.testing_name[count])+'.jpg', g_image[count])
				#saver.save(self.sess, "Model8/model.ckpt")#best:6
	def test(self):
		#if not self.args.train:
		saver = tf.train.Saver()
		saver.restore(self.sess, "trained_model/model.ckpt")
		'''
		tag = {}
		with open('data/sample_testing_text.txt') as testing_file:
			for line in testing_file:
				input_buffer = [i.strip() for i in re.split(',',line.strip())]
				tag[int(input_buffer[0])] = input_buffer[1]

		#print(sorted(tag.items()))
		tag = collections.OrderedDict(sorted(tag.items()))
		#for k,v in tag.items():
		#	print(v,end='\t')
		vec_features = Sent2Vec(tag)
		vec_features = np.array(vec_features)
		print(vec_features.shape)
		
		bz = z_sampler(int(vec_features.shape[0]), self.z_dim)
		feed_dict={self.z : bz, self.right_text : vec_features[:,1:]}
		enc_vec, g_image = self.sess.run([self.right_enc_text, self.x_], feed_dict=feed_dict)
		for i in range(int(vec_features.shape[0])):
			imsave('testing/'+'sample_'+str(i+1)+'_'+str(1)+'.jpg', g_image[i])

		#print(vec_features)
		'''
		'''
		for i in range(int(vec_features.shape[0])):
			for j in range(1):
				bz = z_sampler(1, self.z_dim)
				#print(bz[:10])
				reshaped_vec = vec_features[i,1:].reshape((-1,2400))
				#print(reshaped_vec.shape)
				print(vec_features[i,1:11])
				print(reshaped_vec[0,0:10])
				#test_col = random.randint(0,30000)
				#reshaped_vec = right_training_text[test_col,1:].reshape((-1,2400))
				#print(reshaped_vec[0,:10])
				#set(a).intersection(b)
				#print(reshaped_vec.shape)
				feed_dict={self.z : bz, self.right_text : reshaped_vec}
				enc_vec, g_image = self.sess.run([self.right_enc_text, self.x_], feed_dict=feed_dict)
				#print(enc_vec[0,:10])
				imsave('testing/'+'sample_'+str(i+1)+'_'+str(j+1)+'.jpg', g_image[0])
		'''
		self.testing_tag=[]
		self.testing_name=[]
		with open(self.argv[0]) as testing_file:
			for line in testing_file:
				input_buffer = [i.strip() for i in re.split(',',line.strip())]
				buf = input_buffer[1].split(' ')
				self.testing_name.append(input_buffer[1])
				self.testing_tag.append(self.choose_tags(self.color_hair.index(buf[0]+' '+buf[1]), self.color_eyes.index(buf[2]+' '+buf[3])))
		self.testing_tag=np.array(self.testing_tag)
		'''
		fix_seed_1=[4000, 5000, 75, 150, 400, 1800]
		fix_seed_2=[150, 76, 20, 400, 1800]#fix_seed_2=[76, 161, 150, 131, 141]
		fix_seed_3=[1000, 150, 170, 45, 550]
		'''
		fix_seed_1=[4000, 5000, 75, 150, 400, 1800]
		fix_seed_2=[150, 76, 20, 400, 1800]#fix_seed_2=[76, 161, 150, 131, 141]
		fix_seed_3=[1000, 150, 170, 45, 550]
		
		rand_seed={1:fix_seed_1,2:fix_seed_2,3:fix_seed_3}
		for i in range(5):
			#np.random.seed(int(2200/(i+1)))
			#bz = z_sampler(int(len(self.testing_tag)), self.z_dim)
			#feed_dict={self.z : bz, self.right_text : self.testing_tag}
			#g_image = self.sess.run(self.x_, feed_dict=feed_dict)
			for count in range(int(len(self.testing_tag))):
				np.random.seed(rand_seed[count+1][i])
				bz = z_sampler(1, self.z_dim)
				feed_dict={self.z : bz, self.right_text : self.testing_tag[count].reshape((1,25))}
				g_image = self.sess.run(self.x_, feed_dict=feed_dict)
				imsave('samples/'+'sample_'+str(count+1)+'_'+str(i+1)+'.jpg', g_image[0])
if __name__ == '__main__':
	#parser = argparse.ArgumentParser('')
	#parser.add_argument('--train', action='store_true', help='whether train GAN')
	#parser.add_argument('--test', action='store_true', help='whether test GAN')
	#args = parser.parse_args()

	print('loading model')
	G_Net = Generator()
	D_Net = Discriminator()
	Embedding = Encode_text()
	print('finish loading model')
	'''
	if args.train:
		print('loading training data')
		right_training_image = np.load('training/image_training.npy')
		right_training_text = np.load('training/od_tag.npy')
		#right_training_text = np.load('training/text_training.npy')
		#wrong_training_image = np.load('training/wrong_od_image.npy')
		#wrong_training_text = np.load('training/wrong_text_training.npy')
		str_tag	 = np.load('training_tag_str.npy')
		print('right_training_image.shape : ', right_training_image.shape)
		print('right_training_text.shape : ', right_training_text.shape)
		#print('wrong_training_image.shape : ', wrong_training_image.shape)
		#print('wrong_training_text.shape : ', wrong_training_text.shape)
		print('finishing loading training data')
		d_bg = D_batch_gen(right_training_image, right_training_text, str_tag)
		g_bg = G_batch_gen(right_training_image, right_training_text, str_tag)
		print('Build GAN')
		GAN = WDCGAN(args, G_Net, D_Net, Embedding, g_bg, d_bg)
	'''
	#else:
	print('Build GAN')
	GAN = WDCGAN(sys.argv[1:], G_Net, D_Net, Embedding)