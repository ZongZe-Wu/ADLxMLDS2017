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

def batch_generation(features, labels, batch_size):
	len_size = len(features[0])
	for i in range(0, len_size, batch_size):
		x = features[1][i:i+batch_size]
		y = [labels.get(features[0][j], None)[0][0] for j in range(i,i+len(x))]
		seq_size = [labels.get(features[0][j], None)[1][0] for j in range(i,i+len(x))]
		#print(y.shape)
		yield x, y, seq_size

def seq_padding(label_caption, max_length, max_voc_size):
	#print(label_caption.shape)
	#print([[0]*max_voc_size]*(max_length - len(label_caption)))
	#print((list(label_caption)))
	return np.array((label_caption.tolist() + [[0]*max_voc_size]*(max_length - len(label_caption))))


def max_value(a, b):
	if a >= b:
		return a
	else:
		return b

class Model(object):
	def __init__(self, frame_size, input_size, output_size, batch_szie):
		self.frame_size = frame_size
		self.input_size = input_size
		self.output_size = output_size
		self.batch_szie = batch_szie

		with tf.name_scope('inputs'):
			self.xs = tf.placeholder(tf.float32, shape = [None, self.frame_size, self.input_size], name = 'xs')
			self.ys = tf.placeholder(tf.int32, shape = [None, self.frame_size], name = 'ys')
			self.seq_len = tf.placeholder(tf.int32, shape = [None], name = 'seq_len')
		with tf.variable_scope('LSTM_cell'):
			self.add_cell()
		with tf.name_scope('output_layer'):
			self.add_output_layer()
		with tf.name_scope('cost'):
			self.computing_cost()
		#w_init = tf.random_normal_initializer(0., .1)
		#b_init = tf.constant_initializer(0.1)
		with tf.name_scope('training'):
			self.train_op = tf.train.AdamOptimizer(0.003).minimize(self.losses)

	def add_cell(self):
		lstm_cell_layer = [tf.contrib.rnn.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple = True) for size in [128, 128]] 
		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell_layer)
		self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(cell = multi_rnn_cell,
			inputs = self.xs, dtype=tf.float32,
			time_major = False
		)
		#
		#cell_ouputs size : [batch_size, max_time. depth]
	def add_output_layer(self):
		layer_output_0 = tf.reshape(self.cell_outputs, [-1, 128], name = '0_2D')

		with tf.name_scope('output'):
			layer_output_1 = tf.layers.dense(inputs = layer_output_0, units = 512, activation=tf.nn.relu)
			layer_output_2 = tf.layers.dense(inputs = layer_output_1, units = self.output_size, activation=tf.nn.relu)
			self.logits = tf.nn.softmax(tf.reshape(layer_output_2, [-1, self.frame_size, self.output_size]))
			self.prediction = tf.argmax(self.logits, 2)
	def computing_cost(self):
		self.onehot_label = tf.one_hot(self.ys, depth = self.output_size, axis = -1)
		'''
		self.losses = tf.contrib.seq2seq.sequence_loss(
			self.logits,
			self.ys,
			tf.sequence_mask(self.seq_len, dtype=tf.float32),
			name = 'losses'
		)
		'''
		self.losses = tf.losses.softmax_cross_entropy(onehot_labels = self.onehot_label, logits=self.logits)





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
	vectorizer = CountVectorizer(tokenizer=TB().tokenize)
	vectorizer.fit(list(itertools.chain.from_iterable(caption_all)))
	print(vectorizer.transform(['we']))
	#print(vectorizer.vocabulary_)
	inv_map = {v: k for k, v in vectorizer.vocabulary_.items()}
	inverse = [(value, key) for key, value in vectorizer.vocabulary_.items()]
	max_voc_size = max(inverse)[0]+1
	print(inverse[0])
	print(max_voc_size)
	buf_max = 0
	for i in range(len(label_data)):#len(label_data)
		label_buf = []
		seq_len_buf = []
		for j in range(len(label_data[i]['caption'])):
			#buf = label_data[i]['caption'][j].split(' ')
			#buf = re.split('\W+', label_data[i]['caption'][j])
			#buf = list(filter(str.strip, buf))
			buf = TB().tokenize(label_data[i]['caption'][j])
			buf_max = max_value(buf_max, len(buf))
			arr = find(vectorizer.transform(buf))[1]
			arr1 = np.append(np.array([6086]*80), arr)
			seq_len_buf.append(arr1.shape[0])
			#print(arr.shape)
			label_buf.append(np.append(arr1, np.array([6086]*(50-arr.shape[0]))))
		#print(np.array(label_buf).shape)
		label[label_data[i]['id']] = (np.array(label_buf), seq_len_buf)
	print('filename_list[0] : ', filename_list[0])
	print('len(label[filename_list[0]]) : ', label[filename_list[0]][0].shape)
	print(buf_max)
	#print(vectorizer.stop_words_)
	#print(find(vectorizer.transform(label[filename_list[0]][0]))[1])
	print(vectorizer.inverse_transform(vectorizer.transform(['.']))[0])
	#print(vectorizer.inverse_transform(vectorizer.transform(['A'])[0].todense()[0]))


	sess = tf.Session()
	#frame_size, input_size, output_size, batch_szie
	model = Model(130, 4096, 6087, 64)
	writer = tf.summary.FileWriter("logs", sess.graph)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for i in range (2):
		print('epoch : ', i )
		for train_x, train_y, seq_length in batch_generation(features, label, 64):
			#print('np.array(train_x).shape : ', np.array(train_x).shape)
			#print('np.array(train_y).shape : ', np.array(train_y).shape)
			#print('np.array(seqlen).shape : ', np.array(seq_length).shape)
			#print(seq_length)
			feed_dict = {
				model.xs : train_x,
				model.ys : train_y,
				model.seq_len : seq_length
			}
			_, loss = sess.run([model.train_op, model.losses], feed_dict=feed_dict)
		print('loss : ', loss)
		if i % 10 == 0:
			
			save_path = saver.save(sess, "model.ckpt")
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





if __name__ == '__main__':
	main(sys.argv[1:])	