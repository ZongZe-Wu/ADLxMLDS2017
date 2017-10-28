import numpy as np 
import sys
import zipfile
import re
import time
from sklearn import preprocessing
#from rnn_train import LSTMRNN
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

BATCH_SIZE = 50
CELL_SIZE = 256
LR = 0.008
INPUT_DIM = 69
LABEL_DIM = 48
EPOCHES = 200
MAX_FRAME_LENGTH = 777

def validation(prediction, y_test, frame_amount_test):
	amount = prediction.shape[0]
	#print('prediction.shape : ', prediction.shape, 'y_test.shape : ', y_test.shape)
	frame_amount = 0
	acc = 0
	#acc = np.count_nonzero(prediction - y_test)
	for i in range(amount):
		acc += frame_amount_test[i] - np.count_nonzero(prediction[i,:frame_amount_test[i]]-y_test[i,:frame_amount_test[i]])
		frame_amount += frame_amount_test[i] 
	print('!!!!!!!!!!accuracy : ', round((acc/frame_amount),4))

def batch_generation(features, labels, sequence_size, batch_size):
	len_size = features.shape[0]
	for i in range(0, len_size, batch_size):
		x = features[i:i+batch_size]
		y = labels[i:i+batch_size]
		seq_size = sequence_size[i:i+batch_size]
		yield x, y, seq_size
def batch_padding(input_feature, label, one_hot_label, sequence_size, max_length, input_dim):
	input_size = sequence_size.shape[0]
	sum_of_seq = 0
	#sil array:
	sil_array = np.zeros((48))
	sil_array[36] = 1
	x = np.array([]).reshape((0, INPUT_DIM))
	x = []
	y_one_hot = np.array([]).reshape((0, LABEL_DIM))
	y_one_hot = []
	y = np.array([]).reshape((0, max_length))
	y = []
	sample = 1000
	for i in range(0, input_size, sample):
		#max_length = np.amax(sequence_size[i:i + sample])
		#y = np.empty((max_length,), float32oat)
		#print(i)
		count = int(i/sample)
		for j in range(sample):
			#print(j)
			if count*sample + j >= input_size:
				#b_s = 45
				break
			#else:
				#b_s = sample
			#print('i : ', i, 'j : ', j, 'sum_of_seq : ', sum_of_seq)
			x1 = input_feature[sum_of_seq : sum_of_seq + sequence_size[count*sample + j]]
			y1 = label[sum_of_seq : sum_of_seq + sequence_size[count*sample + j]]
			y1_one_hot = one_hot_label[sum_of_seq : sum_of_seq + sequence_size[count*sample + j]]
			#print('y1.shape', y1.shape)
			#print("np.full(((max_length - x1.shape[0]), ),36) : ", np.full(((max_length - x1.shape[0]), ),36).shape)
			#print('y1_one_hot : ',(y1_one_hot.shape))
			y1_one_hot = np.vstack((y1_one_hot, np.tile(sil_array, ((max_length - y1.shape[0]), 1)) ))
			y1 = np.append(y1, np.full(((max_length - y1.shape[0]), ),36))			
			x1 = np.vstack((x1, np.zeros( ( (max_length - x1.shape[0]), input_dim)) ))
			#print('y1_one_hot : ',(y1_one_hot.shape))
			#seq_size = sequence_size[sum_of_seq : sum_of_seq + sequence_size[i*sample + j]]
			#print('x1.shape', x1.shape)
			#print('y1.shape', y1.shape)
			'''
			if i == 0 and j == 0:
				x = np.vstack([x, x1])
				x = x[np.newaxis,:]
				#y_one_hot = np.vstack([y_one_hot, y1_one_hot])
				#y_one_hot = y_one_hot[np.newaxis,:]
			else:
				x = np.vstack([x, x1[np.newaxis,:]])
				#y_one_hot = np.vstack([y_one_hot, y1_one_hot[np.newaxis,:]])
			'''
			x.append(x1)
			#y = np.vstack([y, y1])
			y.append(y1)
			y_one_hot.append(y1_one_hot) 
			sum_of_seq += sequence_size[count*sample + j]
		#seq_size = sequence_size[count*sample:(count+1)*sample]
	x = np.array(x)
	y = np.array(y)
	y_one_hot = np.array(y_one_hot)
	print('x.shape : ', x.shape, 'y.shape : ', y.shape, 'y_one_hot.shape : ', y_one_hot.shape)
	return x, y, y_one_hot
		#x : [batch_size, max_length, Input_dim]
		#y : [batch_size, Label_dim]
'''		
def batch_padding(input_feature, label, one_hot_label, sequence_size, max_length):
	input_size = sequence_size.shape[0]
	sum_of_seq = 0
	#sil array:
	sil_array = np.zeros((48))
	sil_array[36] = 1
	for i in range(0, input_size, BATCH_SIZE):
		#max_length = np.amax(sequence_size[i:i + BATCH_SIZE])
		x = np.array([]).reshape((0, INPUT_DIM))
		y_one_hot = np.array([]).reshape((0, LABEL_DIM))
		y = np.array([]).reshape((0, max_length))
		#y = np.empty((max_length,), float32oat)
		#print(i)
		count = int(i/50)
		for j in range(BATCH_SIZE):
			if count*BATCH_SIZE + j >= input_size:
				b_s = 45
				break
			else:
				b_s = BATCH_SIZE
			#print('i : ', i, 'j : ', j, 'sum_of_seq : ', sum_of_seq)
			x1 = input_feature[sum_of_seq : sum_of_seq + sequence_size[count*BATCH_SIZE + j]]
			y1 = label[sum_of_seq : sum_of_seq + sequence_size[count*BATCH_SIZE + j]]
			y1_one_hot = one_hot_label[sum_of_seq : sum_of_seq + sequence_size[count*BATCH_SIZE + j]]
			#print('y1.shape', y1.shape)
			#print("np.full(((max_length - x1.shape[0]), ),36) : ", np.full(((max_length - x1.shape[0]), ),36).shape)
			#print('y1_one_hot : ',(y1_one_hot.shape))
			y1_one_hot = np.vstack((y1_one_hot, np.tile(sil_array, ((max_length - y1.shape[0]), 1)) ))
			y1 = np.append(y1, np.full(((max_length - y1.shape[0]), ),36))			
			x1 = np.vstack((x1, np.zeros( ( (max_length - x1.shape[0]), INPUT_DIM)) ))
			#print('y1_one_hot : ',(y1_one_hot.shape))
			#seq_size = sequence_size[sum_of_seq : sum_of_seq + sequence_size[i*BATCH_SIZE + j]]
			#print('x1.shape', x1.shape)
			#print('y1.shape', y1.shape)
			if j == 0:
				x = np.vstack([x, x1])
				x = x[np.newaxis,:]
				y_one_hot = np.vstack([y_one_hot, y1_one_hot])
				y_one_hot = y_one_hot[np.newaxis,:]
			else:
				x = np.vstack([x, x1[np.newaxis,:]])
				y_one_hot = np.vstack([y_one_hot, y1_one_hot[np.newaxis,:]])
			y = np.vstack([y, y1])
			sum_of_seq += sequence_size[count*BATCH_SIZE + j]
		seq_size = sequence_size[count*BATCH_SIZE:(count+1)*BATCH_SIZE]
		yield x, y, y_one_hot, seq_size, b_s
		#x : [batch_size, max_length, Input_dim]
		#y : [batch_size, Label_dim]
'''

def convert(sequence):
	for item in sequence:
		try:
			yield float(item)
		except ValueError as e:
			yield item

def main(argv):
	#zip_file = zipfile.ZipFile(argv[0] + 'mfcc.zip', 'r')
	#print(zip_file.namelist())
	timestamp1 = time.time()
	first_map = []
	second_map = []
	# Read 48_39.map #
	with open(argv[0] + '48_39.map') as myfile:
		for line in myfile:
			input_buffer = line.strip().split('\t')
			first_map.append(input_buffer)
	first_map = np.array(first_map)
	print('first_map', first_map.shape)

	# Read 48phone_char.map #
	with open(argv[0] + '48phone_char.map') as myfile:
		for line in myfile:
			input_buffer = line.strip().split('\t')
			second_map.append(input_buffer)
	second_map = np.array(second_map)
	print('second_map', second_map.shape)

	# Read features for MFCC#
	mfcc_train_data = []
	frame_id = []
	frame_amount = []
	count = 0
	r = re.compile("([a-zA-Z]+)([0-9]+)")
	with zipfile.ZipFile(argv[0] + 'mfcc.zip') as myzip:
		with myzip.open('mfcc/train.ark') as myfile:
			for line in myfile:
				line = line.decode('utf8').strip()
				input_buffer = line.strip().split()
				id_value = input_buffer[0].split('_')
				input_buffer[0] = r.match(id_value[1]).group(2)
				mfcc_train_data.append(input_buffer)
				frame_id.append(id_value[2])
				if id_value[2] == '1':
					if count != 0:
						frame_amount.append(count)
					count = 1
				else:
					count += 1
			print('success')
	mfcc_train_data = np.array(mfcc_train_data).astype(np.float)
	mfcc_train_data = mfcc_train_data[mfcc_train_data[:,0].argsort(kind='mergesort')]
	frame_id = np.array(frame_id)
	frame_amount = np.array(frame_amount)
	print(mfcc_train_data.shape)
	#print(mfcc_train_data[0],'\n',mfcc_train_data[1])
	#print('frame_id_shape',frame_id.shape)
	#print('frame_amount_shape',len(frame_amount))
	#Max_frame_length = np.amax(frame_amount)
	print('frame_amount max : ', MAX_FRAME_LENGTH)
	#mfcc_train_data[mfcc_train_data[:,0].argsort()]
	'''	
	#Read features for fbank#
	fbank_train_data = []
	with zipfile.ZipFile(argv[0] + 'fbank.zip') as myzip:
		with myzip.open('fbank/train.ark') as myfile:
			for line in myfile:
				line = line.decode('utf8').strip()
				input_buffer = line.strip().split()
				id_value = input_buffer[0].split('_')
				input_buffer[0] = r.match(id_value[1]).group(2)
				fbank_train_data.append(input_buffer)
			print('success')
	fbank_train_data = np.array(fbank_train_data)
	print(fbank_train_data.shape)
	print(fbank_train_data[0])
	'''
	# Read labels #
	label = []
	with open(argv[0] + 'train.lab') as myfile:
		for line in myfile:
			input_buffer = line.strip().split(',')
			id_value = input_buffer[0].split('_')
			input_buffer[0] = r.match(id_value[1]).group(2)
			input_buffer[1] = np.where(first_map[:,0] == input_buffer[1])[0][0]
			label.append(input_buffer)
		print('success')
	#new = [list(convert(sublist)) for sublist in label]
	#label = np.array(new,dtype = object)
	label = np.array(label).astype(np.float)

	print(label[0])
	label = label[label[:,0].argsort(kind='mergesort')]
	print(label.shape)
	print(label[0])

	lb = preprocessing.LabelBinarizer()
	lb.fit(label[:,1])
	#print('lb.classes_: ', lb.classes_)
	one_hot_label = lb.transform(label[:,1])
	print('one_hot_label.shape: ', one_hot_label.shape)
	'''
	reordered_label = []
	for i in range(label.shape[0]):
		index = mfcc_train_data[i][0]
		for x in range(label.shape[0]):
			if label[x][0] == index:
				reordered_label.append(label[x:x + frame_amount[i]])
				break
		i = i + frame_amount[i]
	reordered_label = np.array(reordered_label)
	print('reordered_label.shape', reordered_label.shape)
	print(reordered_label[0])
	'''
	
	timestamp2 = time.time()
	x, y, y_one_hot = batch_padding(mfcc_train_data[:,1:], label[:,1], frame_amount, MAX_FRAME_LENGTH)
	print('after padding----')
	X_train, X_test, y_train, y_test, frame_amount_train, frame_amount_test = train_test_split(x, y, frame_amount, test_size=0.2, shuffle = None)
	print('X_train.shape : ', X_train.shape, 'y_train.shape : ', y_train.shape, 'frame_amount_train.shape : ', frame_amount_train.shape)
	print("Data preprocessing took %.2f seconds" % (timestamp2 - timestamp1))

	#========================= Training =========================#
	sess = tf.Session()
	# input_size, output_size, cell_size, batch_size
	model = LSTMRNN(sess, INPUT_DIM, LABEL_DIM, CELL_SIZE, BATCH_SIZE, MAX_FRAME_LENGTH, lb)
	
	#merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs", sess.graph)
	sess.run(tf.global_variables_initializer())
	# $ tensorboard --logdir='logs'

	plt.ion()
	plt.show()
	losses = np.array([])
	iteration = 1
	for i in range(EPOCHES):
		#for train_x, train_y, train_one_hot, train_seq_len, in batch_padding(mfcc_train_data[:,1:], label[:,1], one_hot_label, frame_amount, MAX_FRAME_LENGTH):
		for train_x, train_y, train_seq_len, in batch_generation(X_train, y_train, one_hot_label, frame_amount_train, BATCH_SIZE):
			'''
			print('train_x.shape : ', train_x.shape)
			print('train_y.shape : ', train_y.shape)
			print('train_one_hot : ', train_one_hot.shape)
			'''
			#print('train_seq_len.shape : ',train_seq_len.shape)
			
			#print('train_max_length : ',train_max_length)
			#print(train_seq_len)
			if i == 0:
				feed_dict = {
					model.xs : train_x,
					model.ys : train_y,
					#model.ys_one_hot : train_one_hot,
					model.seq_size : train_seq_len,
					#model.max_length : train_max_length
					# create initial state
				}
			else:
				feed_dict = {
					model.xs : train_x,
					model.ys : train_y,
					#model.ys_one_hot : train_one_hot,
					model.seq_size : train_seq_len,
					#model.max_length : train_max_length,
					#model.cell_init_state : state
					# use last state as the initial state
				}
			_, loss, state, prediction = sess.run(
				[model.train_op, model.losses, model.cell_final_state, model.prediction], feed_dict = feed_dict
			)
			losses = np.append(losses, loss)
			if iteration % 4 == 0:
				print('iterations : ', iteration,'\tlosses : ', losses[-1])
				plt.plot(np.arange(iteration), losses)
				plt.xlabel('iterations')
				plt.ylabel('losses')
				plt.pause(0.1)
				feed_dict = {
					model.xs : X_test,
					model.seq_size : frame_amount_test,
				}
				prediction = sess.run(model.prediction_argmax, feed_dict = feed_dict)
				validation(prediction, y_test, frame_amount_test)
			iteration += 1 
		print('======================epoch finish======================  ', i)
		if i % 5 == 0 and i != 0:
			#print('losses : ', losses)
			#result = sess.run(merged, feed_dict)
			#writer.add_summary(result, i)
			saver = tf.train.Saver()
			save_path = saver.save(sess, 'test' + str(5), global_step=i)
if __name__ == '__main__':
	main(sys.argv[1:])

