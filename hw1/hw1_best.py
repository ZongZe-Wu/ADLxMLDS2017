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
from keras import backend as K
MAX_FRAME_LENGTH = 1000
INPUT_DIM = 69
LABEL_DIM = 48

def remove_noise(answer):
	for i in range(len(answer)):
		
		check = []
		flag = 0
		for j in range(len(answer[i])):
			
			if j == 0:
				check.append(answer[i][j])
			else:
				if answer[i][j] != check[-1]:
					flag += 1
				else:
					flag = 0
			if flag >=3:
				check.append(answer[i][j])
		answer[i] = check
	return answer
			
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

def batch_padding(input_feature, sequence_size, max_length, input_dim):
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
			x1 = np.vstack((x1, np.zeros( ( (max_length - x1.shape[0]), input_dim)) ))
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
	fbank_test_data = []
	mfcc_test_data = []
	frame_id = []
	frame_amount = []
	count = 0
	name_0 = []
	#name_1 = []
	r = re.compile("([a-zA-Z]+)([0-9]+)")
	#with zipfile.ZipFile(argv[0] + 'fbank.zip') as myzip:
	with open(argv[0] + 'fbank/test.ark') as myfile:
		for line in myfile:
			#line = line.decode('utf8').strip()
			input_buffer = line.strip().split()
			id_value = input_buffer[0].split('_')
			#name_0.append(id_value[0]+'_'+id_value[1])
			#input_buffer[0] = r.match(id_value[1]).group(2)
			input_buffer = input_buffer + id_value
			fbank_test_data.append(input_buffer)
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
	#fbank_test_data = np.array(fbank_test_data).astype(np.float)
	fbank_test_data = np.array(fbank_test_data)
	frame_id = np.array(frame_id)
	# Read features for MFCC#
	mfcc_test_data = []
	frame_id = []
	count = 0
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
	count = 1
	for i in range(fbank_test_data.shape[0]-1):
		if fbank_test_data[i,-3] == fbank_test_data[i+1,-3] and fbank_test_data[i,-2] == fbank_test_data[i+1,-2]:
			count += 1
		else:
			frame_amount.append(count)
			count = 1
			name_0.append(fbank_test_data[i,-3]+'_'+fbank_test_data[i,-2])
	frame_amount.append(count)
	name_0.append(fbank_test_data[-1,-3]+'_'+fbank_test_data[-1,-2])
	frame_amount = np.array(frame_amount)
	print('fbank_test_data.shape : ', fbank_test_data.shape)
	print('frame_amount : ', np.sum(frame_amount) - fbank_test_data.shape[0])
	print('name_0 : ', len(name_0), name_0[0])
	#print('name_1 : ', len(name_1))
	test_data1 = batch_padding(fbank_test_data[:,1:-3].astype(np.float), frame_amount, MAX_FRAME_LENGTH, 69)
	test_data2 = batch_padding(mfcc_test_data[:,1:-3].astype(np.float), frame_amount, MAX_FRAME_LENGTH, 39)
	timestamp2 = time.time()
	print("Data preprocessing took %.2f seconds" % (timestamp2 - timestamp1))
	filepath1 = 'my_model-fbank-cnn.hdf5'
	model1 = load_model(filepath1)
	'''
	tt = 23
	three_nine_phone = first_map[tt, 1]
	index = np.where(second_map[:,0] == three_nine_phone)
	ans = second_map[index[0][0], 2]
	print('~~~~~~~~~~',ans)
	'''
	prediction1 = model1.predict(test_data1, batch_size = 128, verbose=1)
	filepath2 = 'my_model-2119.hdf5'
	model2 = load_model(filepath2)
	prediction2 = model2.predict(test_data2, batch_size = 128, verbose=1)
	#prediction = prediction1*0.4 + prediction2*0.6
	print('prediction2.shape : ', prediction2.shape)
	#print(' sum : ',np.sum(frame_amount))
	#print(prediction[0,0,:])

	prediction = []
	for i in range(prediction1.shape[0]):
		for j in range(1000):
			prediction.append(np.append(prediction1[i,j,:], prediction2[i,j,:]))
	prediction = np.array(prediction)
	print('prediction.shape : ', prediction.shape)

	filepath3 = 'my_model-fusing.hdf5'
	model3 = load_model(filepath3)
	prediction3 = model3.predict(prediction, batch_size = 128, verbose=1)
	print('prediction3.shape : ',prediction3.shape)
	prediction3 = prediction3.reshape(frame_amount.shape[0],1000,48)
	print('prediction3.shape : ',prediction3.shape)

	filepath4 = 'my_model-cnn-109.hdf5'
	model4 = load_model(filepath4)
	prediction4 = model4.predict(test_data2, batch_size = 128, verbose=1)

	filepath5 = 'my_model-fbank-118.hdf5'
	model5 = load_model(filepath5)
	prediction5 = model5.predict(test_data1, batch_size = 128, verbose=1)

	prediction6 = (prediction3+prediction1+prediction2+prediction4+prediction5)/5
	answer = []
	for i in range(frame_amount.shape[0]):
		skip = 0
		for j in range(frame_amount[i]):
			#if prediction6[i,j,np.argmax(prediction6[i,j,:])] > 0.5:
			three_nine_phone = first_map[np.argmax(prediction6[i,j,:]), 1]
				#print(three_nine_phone)
			index = np.where(second_map[:,0] == three_nine_phone)
				#index = np.where(second_map[np.argmax(prediction[i,j]),0] == three_nine_phone)
				#print(index)
			ans = second_map[index[0][0], 2]
				#ans = second_map[np.argmax(prediction[i,j]), 2]
			answer.append(ans)
			#else:
			#	skip+=1
		#frame_amount[i]-=skip
	print('answer_length : ', len(answer), answer[0])
	reshape_answer = []
	seq_len = 0
	for i in range(frame_amount.shape[0]):
		reshape_answer.append(answer[seq_len:seq_len+frame_amount[i]])
		seq_len += frame_amount[i]
	print(len(reshape_answer[1]))
	#print(reshape_answer[0])
	reshape_answer = remove_noise(reshape_answer)
	#reshape_answer = remove_single(reshape_answer)
	#reshape_answer = remove_single(reshape_answer)
	#reshape_answer = remove_single(reshape_answer)
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
	K.clear_session()
if __name__ == '__main__':
	main(sys.argv[1:])