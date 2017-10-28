import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Masking, Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from preprocess_data import batch_padding 
from sklearn.model_selection import train_test_split
import time
import sys
import re
import zipfile
import itertools
MAX_FRAME_LENGTH = 1000
INPUT_DIM = 39
LABEL_DIM = 48

def delete_double(answer):
	for i in range(len(answer)):
		answer[i] = list(answer for answer,_ in itertools.groupby(answer[i]))
	return answer
def remove_sil(answer):
	for i in range(len(answer)):
		if answer[i][0] == 'L':
			del answer[i][0]
		if answer[i][-1] == 'L':
			del answer[i][-1]	
	return answer

def batch_padding(input_feature, sequence_size, max_length):
	input_size = sequence_size.shape[0]
	sum_of_seq = 0	
	x = []
	sample = 1000
	for i in range(0, input_size, sample):
		count = int(i/sample)
		for j in range(sample):
			if count*sample + j >= input_size:
				break
			x1 = input_feature[sum_of_seq : sum_of_seq + sequence_size[count*sample + j]]
			x1 = np.vstack((x1, np.zeros( ( (max_length - x1.shape[0]), INPUT_DIM)) ))
			x.append(x1)
			sum_of_seq += sequence_size[count*sample + j]
	x = np.array(x)
	print('x.shape : ', x.shape)
	return x

def main(argv):
	timestamp1 = time.time()
	first_map = []
	second_map = []
	# Read 48_39.map #
	with open(argv[0] + 'phones/48_39.map') as myfile:
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
	mfcc_test_data = []
	frame_id = []
	frame_amount = []
	count = 0
	name_0 = []
	#name_1 = []
	r = re.compile("([a-zA-Z]+)([0-9]+)")
	#with zipfile.ZipFile(argv[0] + 'mfcc.zip') as myzip:
	with open(argv[0] + 'mfcc/test.ark') as myfile:
		for line in myfile:
			#line = line.decode('utf8').strip()
			input_buffer = line.strip().split()
			id_value = input_buffer[0].split('_')
			#name_0.append(id_value[0]+'_'+id_value[1])
			#input_buffer[0] = r.match(id_value[1]).group(2)
			input_buffer = input_buffer + id_value
			mfcc_test_data.append(input_buffer)
			frame_id.append(id_value[2])
			'''
			if id_value[2] == '1':
				if count != 0:
					frame_amount.append(count)
					name_1.append(id_value[0]+'_'+id_value[1])
				count = 1
			else:
				count += 1
			'''
		print('success')
		#frame_amount.append(count)
		#name_1.append(id_value[0]+'_'+id_value[1])
	#mfcc_test_data = np.array(mfcc_test_data).astype(np.float)
	mfcc_test_data = np.array(mfcc_test_data)
	frame_id = np.array(frame_id)

	count = 1
	for i in range(mfcc_test_data.shape[0]-1):
		if mfcc_test_data[i,-3] == mfcc_test_data[i+1,-3] and mfcc_test_data[i,-2] == mfcc_test_data[i+1,-2]:
			count += 1
		else:
			frame_amount.append(count)
			count = 1
			name_0.append(mfcc_test_data[i,-3]+'_'+mfcc_test_data[i,-2])
	frame_amount.append(count)
	name_0.append(mfcc_test_data[-1,-3]+'_'+mfcc_test_data[-1,-2])
	frame_amount = np.array(frame_amount)
	print('mfcc_test_data.shape : ', mfcc_test_data.shape)
	print('frame_amount : ', np.sum(frame_amount) - mfcc_test_data.shape[0])
	print('name_0 : ', len(name_0), name_0[0])
	#print('name_1 : ', len(name_1))
	test_data = batch_padding(mfcc_test_data[:,1:-3].astype(np.float), frame_amount, MAX_FRAME_LENGTH)
	timestamp2 = time.time()
	print("Data preprocessing took %.2f seconds" % (timestamp2 - timestamp1))
	filepath = 'my_model-2119.hdf5'
	model = load_model(filepath)
	'''
	tt = 23
	three_nine_phone = first_map[tt, 1]
	index = np.where(second_map[:,0] == three_nine_phone)
	ans = second_map[index[0][0], 2]
	print('~~~~~~~~~~',ans)
	'''
	prediction = model.predict(test_data, batch_size = 128, verbose=1)
	print('prediction.shape : ', prediction.shape)
	#print(' sum : ',np.sum(frame_amount))
	print(prediction[0,0,:])
	answer = []
	for i in range(frame_amount.shape[0]):
		for j in range(frame_amount[i]):
			three_nine_phone = first_map[np.argmax(prediction[i,j]), 1]
			#print(three_nine_phone)
			index = np.where(second_map[:,0] == three_nine_phone)
			#index = np.where(second_map[np.argmax(prediction[i,j]),0] == three_nine_phone)
			#print(index)
			ans = second_map[index[0][0], 2]
			#ans = second_map[np.argmax(prediction[i,j]), 2]
			answer.append(ans)
	print('answer_length : ', len(answer), answer[0])
	reshape_answer = []
	seq_len = 0
	for i in range(frame_amount.shape[0]):
		reshape_answer.append(answer[seq_len:seq_len+frame_amount[i]])
		seq_len += frame_amount[i]
	print(len(reshape_answer[1]))
	#print(reshape_answer[0])
	reshape_answer = delete_double(reshape_answer)
	reshape_answer = remove_sil(reshape_answer)
	
	file = open(argv[1],'w')
	file.write('id,phone_sequence')
	file.write('\n')
	seq_len = 0
	for i in range (len(frame_amount)):
		file.write(name_0[i])
		file.write(',')
		file.write(''.join(reshape_answer[i]))
		file.write('\n')
	file.close()
	
if __name__ == '__main__':
	main(sys.argv[1:])