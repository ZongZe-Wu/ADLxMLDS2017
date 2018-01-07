import tensorflow as tf 
import numpy as np
from ops import *
import cv2

class Encode_text(object):
	def __init__(self):
		self.text_dim = 2400
		self.emb_dim = 128
		self.name='encode_text'
		print(self.name)
	def __call__(self, text, reuse_variables=True):
		with tf.variable_scope(self.name, reuse=reuse_variables):
			W_enc = weight_variables([self.text_dim, self.emb_dim], name = 'w_enc1')
			b_enc = bias_variables([self.emb_dim], name = 'b_enc1')
			self.encoding_text = tf.nn.relu(tf.matmul(text, W_enc) + b_enc)
		return self.encoding_text
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

class Generator(object):
	def __init__(self, training=True):
		self.z_dim = 100
		self.name='Generation_Network'
		print(self.name)
		#self.batch_size = 64
		# self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		# self.x = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x')
		# self.text = tf.placeholder(tf.float32, [None, 2400], name='text')
		# self._encode_text()
		# self.Gz = self._build_Gnet()
		# self.Dg = self._build_Dnet()
		# self.Dx = self._build_Dnet(reuse_variables=False)
		
		# self.sess = tf.Session()
		# if training:
		# 	self.right_training_image = np.load('image_training.npy')
		# 	self.right_training_text = np.load('text_training.npy')
		# 	self.wrong_training_image = np.load('wrong_od_image.npy')
		# 	self.wrong_training_text = np.load('wrong_text_training.npy')
		# 	self.sess.run(tf.global_variables_initializer())
	# def test_fake_image(self):
	# 	return self.sess.run(self.Gz, feed_dict={self.z : self.z_sampler(1, self.z_dim), self.text : self.z_sampler(1, 2400)})
	# def z_sampler(self, batch_size, z_dim):
	# 	return np.random.normal(0, 1, size=[batch_size, z_dim])
	# def _encode_text(self):
		# with tf.variable_scope('encode_text'):
		# 	W_enc = self._weight_variables([2400, 128], name = 'w_conv1')
		# 	b_enc = self._bias_variables([128], name = 'b_conv1')
		# 	self.encoding_text = tf.matmul(self.text, W_enc) + b_enc
	def __call__(self, z, encoding_text):
		with tf.variable_scope(self.name):
			with tf.variable_scope('concat'):
				self.concat_input = tf.concat([z, encoding_text], 1)
				#tf.shape(self.concat_input)[0]
			with tf.variable_scope('fully_connected'):
				W_fc1 = weight_variables([225, 4*4*1024], name = 'W_fc1')
				b_fc1 = bias_variables([4*4*1024], name = 'b_fc1')
				fc = tf.matmul(self.concat_input, W_fc1) + b_fc1
				reshape_fc = tf.reshape(fc, (tf.shape(self.concat_input)[0], 4, 4, 1024))
				conv1 = relu_batch_norm(reshape_fc)

			with tf.variable_scope('conv2'):
				W_conv2 = weight_variables([4, 4, 512, 1024], name = 'W_conv2')
				b_conv2 = bias_variables([512], name = 'b_conv2')
				h_conv2 = conv2d_transpose(conv1, W_conv2, 2, [tf.shape(self.concat_input)[0], 8, 8, 512]) + b_conv2
				conv2 = relu_batch_norm(h_conv2)
				#print('conv2.shape : ', conv2.shape)
			with tf.variable_scope('conv3'):
				W_conv3 = weight_variables([4, 4, 256, 512], name = 'W_conv3')
				b_conv3 = bias_variables([256], name = 'b_conv3')
				h_conv3 = conv2d_transpose(conv2, W_conv3, 2, [tf.shape(self.concat_input)[0], 16, 16, 256]) + b_conv3
				conv3 = relu_batch_norm(h_conv3)
				#print('conv3.shape : ', conv3.shape)
			with tf.variable_scope('conv4'):
				W_conv4 = weight_variables([4, 4, 128, 256], name = 'W_conv4')
				b_conv4 = bias_variables([128], name = 'b_conv4')
				h_conv4 = conv2d_transpose(conv3, W_conv4, 2, [tf.shape(self.concat_input)[0], 32, 32, 128]) + b_conv4
				conv4 = relu_batch_norm(h_conv4)
				#print('conv4.shape : ', conv4.shape)
			with tf.variable_scope('conv5'):
				W_conv5 = weight_variables([4, 4, 3, 128], name = 'W_conv5')
				b_conv5 = bias_variables([3], name = 'b_conv5')
				h_conv5 = conv2d_transpose(conv4, W_conv5, 2, [tf.shape(self.concat_input)[0], 64, 64, 3]) + b_conv5
				conv5 = tf.nn.sigmoid(h_conv5)	
				#conv5 = (conv5+1)/2
				#print('conv5.shape : ', conv5.shape)
		return conv5
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

class Discriminator(object):
	def __init__(self):
		self.name ='Discriminator_Network'
		print(self.name)
	def __call__(self, x, encoding_text, reuse_variables=True):
		with tf.variable_scope(self.name, reuse=reuse_variables):
			#with tf.variable_scope('conv1'):
			W_conv1 = weight_variables([5, 5, 3, 32], name = 'W_conv1')
			b_conv1 = bias_variables([32], name = 'b_conv1')
			h_conv1 = conv2d(x, W_conv1, 2) + b_conv1
			#conv1 = tf.nn.relu(h_conv1)
			conv1 = lrelu(h_conv1)
			# 32
			print('conv1.shape : ', conv1.shape)
			#with tf.variable_scope('conv2'):
			W_conv2 = weight_variables([4, 4, 32, 64], name = 'W_conv2')
			b_conv2 = bias_variables([64], name = 'b_conv2')
			h_conv2 = conv2d(conv1, W_conv2, 2) + b_conv2
			#conv2 = tf.nn.relu(h_conv2)
			conv2 = lrelu(h_conv2)
			print('conv2.shape : ', conv2.shape)
			#with tf.variable_scope('conv3'):
			# 16
			W_conv3 = weight_variables([4, 4, 64, 128], name = 'W_conv3')
			b_conv3 = bias_variables([128], name = 'b_conv3')
			h_conv3 = conv2d(conv2, W_conv3, 2) + b_conv3
			#conv3 = tf.nn.relu(h_conv3)
			conv3 = lrelu(h_conv3)
			print('conv3.shape : ', conv3.shape)
			#with tf.variable_scope('conv4'):
			# 8
			W_conv4 = weight_variables([3, 3, 128, 256], name = 'W_conv4')
			b_conv4 = bias_variables([256], name = 'b_conv4')
			h_conv4 = conv2d(conv3, W_conv4, 2) + b_conv4
			#conv4 = tf.nn.relu(h_conv4)
			conv4 = lrelu(h_conv4)
			print('conv4.shape : ', conv4.shape)
			'''
			flatten = tf.reshape(conv4, (-1, 4*4*64))
			concat_image_text = tf.concat([flatten, encoding_text], -1)
			W_fc1 = weight_variables([4*4*64 + 128, 1], name = 'W_fc1')
			b_fc1 = bias_variables([1], name = 'b_fc1')
			fc1 = tf.matmul(concat_image_text, W_fc1) + b_fc1
			print('fc1.shape : ', fc1.shape)
			'''
			encoding_reshape = tf.reshape(encoding_text,(-1,1,1,25))
			encoding_reshape = tf.tile(encoding_reshape, [1,4,4,1])
			#flatten = tf.reshape(conv4, (-1, 4*4*64))
			#print('conv4.shape : ', conv4.shape)
			#with tf.variable_scope('fc1'):
			#concat_image_text = tf.concat([flatten, encoding_text], -1)
			concat_image_text = tf.concat([conv4, encoding_reshape], 3)
			print('concat_image_text.shape : ',concat_image_text.shape)
			#4*4*(16+8) = 4,4,24
			W_conv5 = weight_variables([4, 4, 256+25, 256], name = 'W_conv5')
			b_conv5 = bias_variables([256], name = 'b_conv5')
			h_conv5 = conv2d(concat_image_text, W_conv5, 2) + b_conv5
			#conv5 = tf.nn.relu(h_conv5)
			conv5 = lrelu(h_conv5)
			print('conv5.shape : ', conv5.shape)
			# 2,2,8
			W_conv6 = weight_variables([4, 4, 256, 1], name = 'W_conv6')
			b_conv6 = bias_variables([1], name = 'b_conv6')
			h_conv6 = conv2d(conv5, W_conv6, 2) + b_conv6
			#conv4 = tf.nn.relu(h_conv4)
			conv6 = h_conv6
			print('conv6.shape : ', conv6.shape)
			flatten = tf.reshape(conv6, (-1,1))
			#W_fc1 = weight_variables([4*4*64 + 128, 1], name = 'W_conv5')
			#b_fc1 = bias_variables([1], name = 'b_fc1')
			#fc1 = tf.matmul(concat_image_text, W_fc1) + b_fc1
		return flatten
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]	

# test = Improved_WDCGAN()
# test_image = test.test_fake_image()
# print(test_image.shape)
# cv2.imshow("image", test_image.reshape(64,64,3))
# cv2.waitKey(1000)