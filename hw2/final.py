import sys 
import os
import numpy as np
import json
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import re
import tensorflow as tf
from token_test import TreebankWordTokenizer as TB
from scipy.sparse import csr_matrix, find
import time
from random import random,randint,seed
def batch_generation(features, labels, batch_size, iteration, loss, prev_rd, max_voc_size):
	len_size = len(features[0])
	#if iteration % 30 :
	#seed(int(iteration/20))
	#rd = randint(0,37)
	#if loss < 20*(500-iteration)/500:
		#seed(int(iteration/20))
	rd = randint(0,37)
	#else:
		#rd = prev_rd
	for i in range(0, len_size, batch_size):
		#if iteration % 2 == 0:		
		x = features[1][i:i+batch_size]
		y = [labels.get(features[0][j], None)[0][rd] if rd < len(labels.get(features[0][j], None)[0]) else labels.get(features[0][j], None)[0][rd%len(labels.get(features[0][j], None)[0])] for j in range(i,i+len(x))]
		seq_size = [labels.get(features[0][j], None)[1][rd] if rd < len(labels.get(features[0][j], None)[1]) else labels.get(features[0][j], None)[1][rd%len(labels.get(features[0][j], None)[1])] for j in range(i,i+len(x))]
		mask = [([0]*80 + [1]*labels.get(features[0][j], None)[1][rd] + [0]*(50-labels.get(features[0][j], None)[1][rd])) if rd < len(labels.get(features[0][j], None)[1]) else ([0]*80 + [1]*labels.get(features[0][j], None)[1][rd%len(labels.get(features[0][j], None)[1])] + [0]*(50-labels.get(features[0][j], None)[1][rd%len(labels.get(features[0][j], None)[1])])) for j in range(i,i+len(x))]
		bos = [max_voc_size+1]*batch_size
		#print('mask len : ', mask)
		prev_rd = rd
		yield x, y, seq_size, np.array(mask), prev_rd, bos

def seq_padding(label_caption, max_length, max_voc_size):
	return np.array((label_caption.tolist() + [[0]*max_voc_size]*(max_length - len(label_caption))))

def max_value(a, b):
	if a >= b:
		return a
	else:
		return b
class Testmodel(object):
	def __init__(self, frame_size, input_size, output_size, training):
		self.frame_size = frame_size
		self.input_size = input_size
		self.output_size = output_size
		#self.batch_size = batch_size
		self.threshold_sampling = 0.5
		self.training = training
		with tf.name_scope('inputs'):
			self.xs = tf.placeholder(tf.float32, shape = [None, self.frame_size, self.input_size], name = 'xs')
			self.ys = tf.placeholder(tf.int32, shape = [None, self.frame_size], name = 'ys')
			self.seq_len = tf.placeholder(tf.int32, shape = [None], name = 'seq_len')
			self.mask = tf.placeholder(tf.float32, shape = [None, self.frame_size], name = 'mask')
			self.batch_size = tf.placeholder(tf.int32, [], name = 'batch_size')
			self.bos = tf.placeholder(tf.int32, [None], name = 'bos')
			self.for_training = tf.placeholder(tf.bool, name = 'for_training')
		with tf.variable_scope('LSTM_cell'):
			self.Ws_embedding = self._weight_variables([self.output_size, 256], name = 'weights0')
			self.bs_embedding = self._bias_variables([1, 256], name = 'biases0')
			self.Ws_decode = self._weight_variables([256, self.output_size], name = 'weights1')
			self.bs_decode = self._bias_variables([1, self.output_size], name = 'biases1')
			self.Watt_decode = self._weight_variables([512, 1], name = 'weightsatt')
			self.batt_decode = self._bias_variables([1, 1], name = 'biasesatt')
			self.add_cell()
		#with tf.name_scope('cost'):
			#self.computing_cost()
		#with tf.name_scope('train'):
			#self.train_op = tf.train.GradientDescentOptimizer(0.001).minimize(self.mean_losses)

	def _weight_variables(self, shape, name):
		initializer = tf.random_normal_initializer(mean = 0., stddev = 1.)
		return tf.get_variable(shape = shape, initializer = initializer, name = name)
	
	def _bias_variables(self, shape, name):
		initializer = tf.constant_initializer(0.1)
		return tf.get_variable(shape = shape, initializer = initializer, name = name)

	def add_cell(self):
		#caption encode
		self.onehot_label = tf.one_hot(self.ys, depth = self.output_size, axis = -1)
		self.onehot_bos = tf.one_hot(self.bos, depth = self.output_size, axis = -1)
		print('self.onehot_label.shape : ', self.onehot_label.shape)
		self.embedding_ys = tf.reshape(tf.matmul(tf.reshape(self.onehot_label, [-1, self.output_size]), self.Ws_embedding) + self.bs_embedding, [-1, self.frame_size, 256])
		self.embedding_bos = tf.reshape(tf.matmul(tf.reshape(self.onehot_bos, [-1, self.output_size]), self.Ws_embedding) + self.bs_embedding, [-1, 256])
		print('self.embedding_ys.shape :', self.embedding_ys.shape)
		print('self.embedding_bos.shape :', self.embedding_bos.shape)
		#encode		
		
		#with tf.variable_scope('lstm'):
		#tf.get_variable_scope().reuse_variables()
		#encoder_layer = [tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 1.0, state_is_tuple = True) for size in [256, 256]]
		encode_layer_0 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias = 1.0, state_is_tuple = True)
		encode_layer_1 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias = 1.0, state_is_tuple = True)
		init_state_0 = encode_layer_0.zero_state(self.batch_size, dtype = tf.float32)
		init_state_1 = encode_layer_1.zero_state(self.batch_size, dtype = tf.float32)
		for i in range(80):
			if i == 0:
				#tf.get_variable_scope().reuse_variables()
				with tf.variable_scope('lstm_0'):
					output_states, internal_state = encode_layer_0(self.xs[:,i,:], init_state_0)
					concat_output_states = tf.concat([output_states, tf.zeros([self.batch_size, 256])], 1)
				with tf.variable_scope('lstm_1'):
					output_states_2, internal_state_2 = encode_layer_1(concat_output_states, init_state_1)
					att_mat = tf.expand_dims(output_states_2, 1)
					#tf.expand_dims(t, 0)
			else:
				tf.get_variable_scope().reuse_variables()
				with tf.variable_scope('lstm_0'):
					output_states, internal_state = encode_layer_0(self.xs[:,i,:], internal_state)
					concat_output_states = tf.concat([output_states, tf.zeros([self.batch_size, 256])], 1)

				with tf.variable_scope('lstm_1'):
					output_states_2, internal_state_2 = encode_layer_1(concat_output_states, internal_state_2)
					att_mat = tf.concat([att_mat, tf.expand_dims(output_states_2, 1)], 1)
			#with tf.variable_scope('lstm_1'):

			decode_layer = tf.matmul(output_states_2, self.Ws_decode) + self.bs_decode

		print('encoding~~~')

		for i in range(80, self.frame_size):

			tf.get_variable_scope().reuse_variables()
			with tf.variable_scope('lstm_0'):		
				output_states, internal_state = encode_layer_0(self.xs[:,i,:], internal_state)
				if i == 80:
					#self.bos = tf.one_hot([6087]*50, depth = self.output_size, axis = -1)
					#self.embedding_bos = tf.reshape(tf.matmul(tf.reshape(self.bos, [-1, self.output_size]), self.Ws_embedding) + self.bs_embedding, [-1, 256])
					fused_input = tf.concat([output_states, self.embedding_bos], 1)
				else:
					fused_input = tf.concat([output_states, output_states_2], 1)
				#fused_input = tf.concat([weighted_sum, output_states, self.schedule_sampling(0.995, i-1, output_states_2)], 1)
			with tf.variable_scope('lstm_1'):
				output_states_2, internal_state_2 = encode_layer_1(fused_input, internal_state_2)		
			#embedding_layer = tf.matmul(output_states_2, self.Ws_embedding) + self.bs_embedding
			decode_layer_1 = tf.matmul(output_states_2, self.Ws_decode) + self.bs_decode
			if i == 80:
				predict = tf.expand_dims(decode_layer_1,1)
			else:
				predict = tf.concat([predict, tf.expand_dims(decode_layer_1, 1)], 1)
			#correct_prediction = tf.equal(tf.argmax(decode_layer,-1), tf.argmax(self.onehot_label[:,i,:],-1))
			#word_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.argmax(predict, 2)
		self.answer = tf.argmax(self.onehot_label, 2)
		self.ground_truth = self.ys
		print('decoding ~~~')

class Model(object):
	def __init__(self, frame_size, input_size, output_size, training):
		self.frame_size = frame_size
		self.input_size = input_size
		self.output_size = output_size
		#self.batch_size = batch_size
		self.threshold_sampling = 0.5
		self.training = training
		with tf.name_scope('inputs'):
			self.xs = tf.placeholder(tf.float32, shape = [None, self.frame_size, self.input_size], name = 'xs')
			self.ys = tf.placeholder(tf.int32, shape = [None, self.frame_size], name = 'ys')
			self.seq_len = tf.placeholder(tf.int32, shape = [None], name = 'seq_len')
			self.mask = tf.placeholder(tf.float32, shape = [None, self.frame_size], name = 'mask')
			self.batch_size = tf.placeholder(tf.int32, [], name = 'batch_size')
			self.bos = tf.placeholder(tf.int32, [None], name = 'bos')
			self.for_training = tf.placeholder(tf.bool, name = 'for_training')
		with tf.variable_scope('LSTM_cell'):
			self.Ws_embedding = self._weight_variables([self.output_size, 256], name = 'weights0')
			self.bs_embedding = self._bias_variables([1, 256], name = 'biases0')
			self.Ws_decode = self._weight_variables([256, self.output_size], name = 'weights1')
			self.bs_decode = self._bias_variables([1, self.output_size], name = 'biases1')
			self.Watt_decode = self._weight_variables([512, 1], name = 'weightsatt')
			self.batt_decode = self._bias_variables([1, 1], name = 'biasesatt')
			self.add_cell()
		with tf.name_scope('cost'):
			self.computing_cost()
		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.mean_losses)

	def _weight_variables(self, shape, name):
		initializer = tf.random_normal_initializer(mean = 0., stddev = 1.)
		return tf.get_variable(shape = shape, initializer = initializer, name = name)
	
	def _bias_variables(self, shape, name):
		initializer = tf.constant_initializer(0.1)
		return tf.get_variable(shape = shape, initializer = initializer, name = name)
	def test(self):
		print(tf.cond(self.for_training, lambda : tf.constant(10), lambda : tf.constant(0)))
	def schedule_sampling(self, decade_rate, caption_index, own_guess):
		return tf.cond(self.for_training, lambda : self.g_truth(caption_index, own_guess), lambda : self.guess(own_guess))
	def guess(self, own_guess):
		#print('fuckkkkkkkk')
		return own_guess
	def g_truth(self, caption_index, own_guess):
		if not self.training:
			return own_guess
		else:
			#self.threshold_sampling = self.threshold_sampling*decade_rate
			if random() <= self.threshold_sampling:
				return self.embedding_ys[:, caption_index]
			else:
				return own_guess

	def add_cell(self):
		#caption encode
		self.onehot_label = tf.one_hot(self.ys, depth = self.output_size, axis = -1)
		self.onehot_bos = tf.one_hot(self.bos, depth = self.output_size, axis = -1)
		print('self.onehot_label.shape : ', self.onehot_label.shape)
		self.embedding_ys = tf.reshape(tf.matmul(tf.reshape(self.onehot_label, [-1, self.output_size]), self.Ws_embedding) + self.bs_embedding, [-1, self.frame_size, 256])
		self.embedding_bos = tf.reshape(tf.matmul(tf.reshape(self.onehot_bos, [-1, self.output_size]), self.Ws_embedding) + self.bs_embedding, [-1, 256])
		print('self.embedding_ys.shape :', self.embedding_ys.shape)
		print('self.embedding_bos.shape :', self.embedding_bos.shape)
		#encode		
		
		#with tf.variable_scope('lstm'):
		#tf.get_variable_scope().reuse_variables()
		#encoder_layer = [tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 1.0, state_is_tuple = True) for size in [256, 256]]
		encode_layer_0 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias = 1.0, state_is_tuple = True)
		encode_layer_1 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias = 1.0, state_is_tuple = True)
		init_state_0 = encode_layer_0.zero_state(self.batch_size, dtype = tf.float32)
		init_state_1 = encode_layer_1.zero_state(self.batch_size, dtype = tf.float32)
		for i in range(80):
			if i == 0:
				#tf.get_variable_scope().reuse_variables()
				with tf.variable_scope('lstm_0'):
					output_states, internal_state = encode_layer_0(self.xs[:,i,:], init_state_0)
					concat_output_states = tf.concat([output_states, tf.zeros([self.batch_size, 256])], 1)
				with tf.variable_scope('lstm_1'):
					output_states_2, internal_state_2 = encode_layer_1(concat_output_states, init_state_1)
					att_mat = tf.expand_dims(output_states_2, 1)
					#tf.expand_dims(t, 0)
			else:
				tf.get_variable_scope().reuse_variables()
				with tf.variable_scope('lstm_0'):
					output_states, internal_state = encode_layer_0(self.xs[:,i,:], internal_state)
					concat_output_states = tf.concat([output_states, tf.zeros([self.batch_size, 256])], 1)

				with tf.variable_scope('lstm_1'):
					output_states_2, internal_state_2 = encode_layer_1(concat_output_states, internal_state_2)
					att_mat = tf.concat([att_mat, tf.expand_dims(output_states_2, 1)], 1)
			#with tf.variable_scope('lstm_1'):
				#decode_layer = tf.layers.dense(inputs = output_states_2, units = self.output_size,activation = tf.nn.relu, name = 'decoding')
			#Ws_decode = self._weight_variables([256, self.output_size], name = 'weights0')
			#bs_decode = self._bias_variables([1, 6087], name = 'biases0')
			#embedding_layer = tf.matmul(output_states_2, self.Ws_embedding) + self.bs_embedding
			decode_layer = tf.matmul(output_states_2, self.Ws_decode) + self.bs_decode
			loss = tf.losses.softmax_cross_entropy(self.onehot_label[:,i,:], decode_layer)
			if i == 0:
				self.losses = loss*self.mask[:,i]
			else:
				self.losses += loss*self.mask[:,i]
			#print('att_mat~~~~~ :', att_mat.shape)
		print('encoding~~~')
		
		#print('self.embedding_bos.shape :', self.embedding_bos.shape)
		#decoder :
		#Ws_decode = self._weight_variables([1024, 1], name = 'weights0')
		#bs_decode = self._bias_variables([1], name = 'biases0')
		for i in range(80, self.frame_size):
			# Attention
			'''
			for j in range(80):
				att_concat_out = tf.concat([att_mat[:,j,:], internal_state_2[1]], -1)
				#print('att_concat_out~~~~~ :', att_concat_out.shape)
				l_0 = tf.matmul(att_concat_out, self.Watt_decode) + self.batt_decode
				#l_0 = tf.layers.dense(inputs = att_concat_out, units = 256, activation=tf.nn.relu, name = 'L_0_')
				#l_0 = (tf.matmul(att_concat_out, Ws_decode) + bs_decode)
				if j == 0:
					l_1 = l_0
				else:
					l_1 = tf.concat([l_1, l_0], 1)
			#print('l1.shape : ', l_1.shape)
			attention_result = tf.nn.softmax(l_1)

			#print('attention_result~~~~~ :', attention_result.shape)
			weighted_sum = tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(attention_result,2),[1,1,256]), att_mat), 1)
			'''
			#print('weighted_sum~~~~~ :', weighted_sum.shape)
			#reshape_out = tf.reshape(tf.tile(output_states_2,[80]), [80,])	
			tf.get_variable_scope().reuse_variables()
			with tf.variable_scope('lstm_0'):		
				output_states, internal_state = encode_layer_0(self.xs[:,i,:], internal_state)
				if i == 80:
					#self.bos = tf.one_hot([6087]*50, depth = self.output_size, axis = -1)
					#self.embedding_bos = tf.reshape(tf.matmul(tf.reshape(self.bos, [-1, self.output_size]), self.Ws_embedding) + self.bs_embedding, [-1, 256])
					fused_input = tf.concat([output_states, self.embedding_bos], 1)
				else:
					fused_input = tf.concat([output_states, self.schedule_sampling(0.99999999, i-1, output_states_2)], 1)
				#fused_input = tf.concat([weighted_sum, output_states, self.schedule_sampling(0.995, i-1, output_states_2)], 1)
			with tf.variable_scope('lstm_1'):
				output_states_2, internal_state_2 = encode_layer_1(fused_input, internal_state_2)		
			#embedding_layer = tf.matmul(output_states_2, self.Ws_embedding) + self.bs_embedding
			decode_layer_1 = tf.matmul(output_states_2, self.Ws_decode) + self.bs_decode
			if i == 80:
				predict = tf.expand_dims(decode_layer_1,1)
			else:
				predict = tf.concat([predict, tf.expand_dims(decode_layer_1, 1)], 1)
			#correct_prediction = tf.equal(tf.argmax(decode_layer,-1), tf.argmax(self.onehot_label[:,i,:],-1))
			#word_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			loss = tf.losses.softmax_cross_entropy(self.onehot_label[:,i,:], decode_layer_1)
			self.losses += loss*self.mask[:,i]
		self.prediction = tf.argmax(predict, 2)
		self.answer = tf.argmax(self.onehot_label, 2)
		self.ground_truth = self.ys
		print('decoding ~~~')

	def computing_cost(self):
		self.mean_losses = tf.reduce_sum(self.losses)/tf.cast(self.batch_size, tf.float32)

def main(argv):
	test = ['klteYv1Uv9A_27_33.avi.npy', '5YJaS2Eswg0_22_26.avi.npy', 'UbmZAe5u5FI_132_141.avi.npy', 'JntMAcTlOF0_50_70.avi.npy', 'tJHUH9tpqPg_113_118.avi.npy']
	#for filename in glob.glob(os.path.join(path, '*.txt')):
	directory = argv[0] + 'training_data/feat/'
	filename_list = []
	feature_list = []
	
	for filename in os.listdir(directory):
		if filename.endswith(".npy"):
			#print(os.path.join(directory, filename))
			feature_list.append(np.vstack((np.load(os.path.join(directory, filename)), np.zeros((50,4096)) ) ) )
			filename_list.append(filename.replace('.npy',''))
		else:
			continue

	feature_arr = np.array(feature_list)
	
	print('feature_arr.shape : ', feature_arr.shape)
	print('len(filename_list) : ', len(filename_list))
	features = (filename_list, feature_arr)
	
	directory = argv[0] + 'testing_data/feat/'
	test_feature_list = []
	test_file = []
	testing_npy = []
	for filename in os.listdir(directory):
		if filename.endswith(".npy"):
			#print(os.path.join(directory, filename))
			test_feature_list.append(np.vstack((np.load(os.path.join(directory, filename)), np.zeros((50,4096)) ) ) )
			test_file.append(filename.replace('.npy',''))
			if filename in test:
				testing_npy.append(np.vstack((np.load(os.path.join(directory, filename)), np.zeros((50,4096)) ) ) )
		else:
			continue
	testing_npy = np.array(testing_npy)
	print('testing_npy : ', testing_npy.shape)		
	test_feature_list = np.array(test_feature_list)
	
	with open(argv[0] + 'training_label.json') as label_file:
		label_data = json.load(label_file)
	#print(label_data[0]['id'])
	label = {}
	caption_all = []
	for i in range(len(label_data)):
		caption_all.append(label_data[i]['caption'])
		#label[label_data[i]['id']] = label_data[i]['caption']
	print(len(caption_all))
	#print(label[filename_list[0]])
	vectorizer = CountVectorizer(tokenizer=TB().tokenize, min_df = 0.0001)
	vectorizer.fit(list(itertools.chain.from_iterable(caption_all)))
	print(vectorizer.transform(['we']))
	#print(vectorizer.vocabulary_)
	
	inverse = [(value, key) for key, value in vectorizer.vocabulary_.items()]
	max_voc_size = max(inverse)[0]+2
	vectorizer.vocabulary_["unknown"] = max_voc_size -1
	inv_map = {v: k for k, v in vectorizer.vocabulary_.items()}
	print('inv_map 0 :', inv_map[0])
	print(inverse[0])
	print("size~~~: ",max_voc_size)
	buf_max = 0
	buf_len_min =100
	buf_len_max =0
	for i in range(len(label_data)):#len(label_data)
		label_buf = []
		seq_len_buf = []
		if buf_len_min >= len(label_data[i]['caption']):
			buf_len_min = len(label_data[i]['caption'])
		if buf_len_max <= len(label_data[i]['caption']):
			buf_len_max = len(label_data[i]['caption'])
		for j in range(len(label_data[i]['caption'])):
			dot_buf = label_data[i]['caption'][j].replace('.','')
			buf = TB().tokenize(dot_buf)
			for k in range(len(buf)):
				if not buf[k] in vectorizer.stop_words_:
					buf[k] = buf[k]
				else:
					buf[k] = "unknown"
			buf_max = max_value(buf_max, len(buf))
			arr = (vectorizer.transform(buf)).nonzero()[1]
			arr1 = np.append(np.array([max_voc_size]*80), arr)
			seq_len_buf.append(arr.shape[0])
			#print(arr.shape)
			label_buf.append(np.append(arr1, np.array([max_voc_size]*(50-arr.shape[0]))))
		#print(np.array(label_buf).shape)
		label[label_data[i]['id']] = (np.array(label_buf), seq_len_buf)
	print('filename_list[0] : ', filename_list[0])
	print('len(label[filename_list[0]]) : ', label[filename_list[0]][0][0])
	print(buf_max)
	print('buf_len_min : ',buf_len_min)
	print('buf_len_max : ', buf_len_max)
	#print(vectorizer.stop_words_)
	#print(find(vectorizer.transform(label[filename_list[0]][0]))[1])
	print(vectorizer.inverse_transform(vectorizer.transform(['.']))[0])
	print(vectorizer.transform(['.']))
	print(inv_map[12])
	print(label_buf[-1])
	print('80~~~~',label_buf[-1][80])
	for j in range(len(label_buf[-1])):
		if label_buf[-1][j] != max_voc_size and label_buf[-1][j] != max_voc_size+1:
			print(inv_map[label_buf[-1][j]], end=' ')
	print('\n')
	#print(vectorizer.inverse_transform(vectorizer.transform(['A'])[0].todense()[0]))

	tf.reset_default_graph()
	sess = tf.Session()	
	#frame_size, input_size, output_size, batch_szie

	model = Model(130, 4096, max_voc_size+1, True)
	writer = tf.summary.FileWriter("logs", sess.graph)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	prev_rd = 0
	mean_losses = 100
	for i in range (5000):
		print('epoch : ', i )
		timestamp1 = time.time()
		avg_accuracy = 0
		iteration = 0
		for train_x, train_y, seq_length, mask, rd, bos in batch_generation(features, label, 50, i, mean_losses, prev_rd, max_voc_size):
			#print('np.array(train_x).shape : ', np.array(train_x).shape)
			#print('np.array(train_y).shape : ', np.array(train_y).shape)
			#print('np.array(seqlen).shape : ', np.array(seq_length).shape)
			#print(seq_length)
			feed_dict = {
				model.xs : train_x,
				model.ys : train_y,
				model.seq_len : seq_length,
				model.mask : mask,
				model.batch_size : int(np.array(train_x).shape[0]),
				model.bos : bos,
				model.for_training : True
			}
			_, mean_losses, prediction, answer, ground_truth = sess.run([model.train_op, model.mean_losses, model.prediction, model.answer, model.ground_truth], feed_dict=feed_dict)
			#avg_accuracy += accuracy
			iteration += 1
			#print('\tmean_losses : ', mean_losses)
		#print(logits[0])
			#print(np.array(prediction).shape)
		timestamp2 = time.time()
		print("%.2f seconds" % (timestamp2 - timestamp1),'\tmean_losses : ', mean_losses)
		prev_rd = rd
		check = randint(0,49)
		for j in range(len(prediction[check])):
			if prediction[check][j] != max_voc_size and prediction[check][j] != max_voc_size+1:
				print(inv_map[prediction[check][j]], end=' ')
			else:
				print(prediction[check][j], end=' ')
		print('\n')
		for j in range(len(answer[check])):
			if answer[check][j] != max_voc_size and answer[check][j] != max_voc_size+1:
				print(inv_map[answer[check][j]], end=' ')
		print('\n')
		test_prediction = sess.run(model.prediction,feed_dict = {model.ys : train_y[:5], model.for_training : False, model.xs : test_feature_list[:5,:,:], model.batch_size : int((test_feature_list[:5,:,:]).shape[0]), model.bos : [max_voc_size+1]*int((test_feature_list[:5,:,:]).shape[0])})
		#print(test)
		print('~~~~fucking testing~~~~~')
		for k in range(5):
			for j in range(len(test_prediction[k])):
				if test_prediction[k][j] != max_voc_size and test_prediction[k][j] != max_voc_size+1:
					print(inv_map[test_prediction[k][j]], end=' ')
				else:
					print(test_prediction[k][j], end=' ')
			else:	
				print(test_prediction[k][j], end=' ')
			print('\n')	
		'''
		for j in range(len(ground_truth[-1])):
			if ground_truth[-1][j] != 6086:
				print(inv_map[ground_truth[-1][j]], end=' ')
		print('\n')
		for j in range(len(train_y[-1])):
			if train_y[-1][j] != 6086:
				print(inv_map[train_y[-1][j]], end=' ')
		print('\n')
		'''
		if i % 5 == 0:	
			print('~~~~~save~~~~~modle~~~~~fuck')		
			save_path = saver.save(sess, "final/final.ckpt")
	'''	
	prediction = sess.run([model.prediction],feed_dict = {model.xs : testing_npy})
	print(np.array(prediction).shape)
	
	file = open(argv[1],'w')
	for i in range(5):
		file.write(test[i])
		file.write(',')
		for j in range(80,100):
			if prediction[0][i][j] == 6086:
				file.write('.')
				file.write(' ')
			else:
				file.write(inv_map[prediction[0][i][j]])
		file.write('\n')

	file.close()
	'''




if __name__ == '__main__':
	main(sys.argv[1:])	