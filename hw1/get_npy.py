import numpy as np 
import sys
import zipfile
import re
import h5py
import time
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Masking, Dropout, Bidirectional, GRU
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from preprocess_data import batch_padding 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tempfile import TemporaryFile

MAX_FRAME_LENGTH = 1000
def main(argv):
	#zip_file = zipfile.ZipFile(argv[0] + 'mfcc.zip', 'r')
	#print(zip_file.namelist())
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
	frame_amount = []
	# Read features for MFCC#
	
	mfcc_train_data = []
	frame_id = []
	#frame_amount = []
	count = 0
	r = re.compile("([a-zA-Z]+)([0-9]+)")
	
	#with zipfile.ZipFile(argv[0] + 'mfcc.zip') as myzip:
	with open(argv[0] + 'mfcc/train.ark') as myfile:
		for line in myfile:
			#line = line.decode('utf8').strip()
			input_buffer = line.strip().split()
			id_value = input_buffer[0].split('_')
			#input_buffer[0] = r.match(id_value[1]).group(2)
			input_buffer = input_buffer + id_value
			mfcc_train_data.append(input_buffer)
			#frame_id.append(id_value[2])
			'''
			if id_value[2] == '1':
				if count != 0:
					frame_amount.append(count)
				count = 1
			else:
				count += 1
			'''
		print('success--mfccd')
		#frame_amount.append(count)
	#mfcc_train_data = np.array(mfcc_train_data).astype(np.float)
	mfcc_train_data = np.array(mfcc_train_data)

	#mfcc_index = mfcc_train_data[:,-3].argsort(kind='mergesort')
	
	#Read features for fbank#
	fbank_train_data = []
	#with zipfile.ZipFile(argv[0] + 'fbank.zip') as myzip:
	with open(argv[0] + 'fbank/train.ark') as myfile:
		for line in myfile:
			#line = line.decode('utf8').strip()
			input_buffer = line.strip().split()
			id_value = input_buffer[0].split('_')
			#input_buffer[0] = r.match(id_value[1]).group(2)
			input_buffer = input_buffer + id_value
			fbank_train_data.append(input_buffer)
		print('success-fbank')
	fbank_train_data = np.array(fbank_train_data)
	print(fbank_train_data.shape)
	print(fbank_train_data[0])
	
	index_1 = mfcc_train_data[:,-3].argsort(kind='mergesort')
	mfcc_train_data = mfcc_train_data[index_1]
	fbank_train_data = fbank_train_data[index_1]
	index_2 = mfcc_train_data[:,-2].argsort(kind='mergesort')
	mfcc_train_data = mfcc_train_data[index_2]
	fbank_train_data = fbank_train_data[index_2]
	print('after sort')
	#print(np.unique(mfcc_train_data[:,0]).shape)
	#frame_id = np.array(frame_id)
	
	count = 1
	for i in range(mfcc_train_data.shape[0]-1):
		if mfcc_train_data[i,-3] == mfcc_train_data[i+1,-3] and mfcc_train_data[i,-2] == mfcc_train_data[i+1,-2]:
			count += 1
		else:
			frame_amount.append(count)
			count = 1
	frame_amount.append(count)
	
	frame_amount = np.array(frame_amount)
	np.save('frame_amount.npy', frame_amount)
	#print(mfcc_train_data[0:100,0])
	print(mfcc_train_data[0,0], mfcc_train_data[1,0])
	#print(fbank_train_data[0,0], fbank_train_data[1,0])
	#print(mfcc_train_data.shape)
	#print(mfcc_train_data[0],'\n',mfcc_train_data[1])
	#print('frame_id_shape',frame_id.shape)
	#print('frame_amount_shape',len(frame_amount))
	#Max_frame_length = np.amax(frame_amount)
	print('frame_amount.shape', frame_amount.shape)
	print(frame_amount[0],frame_amount[-1])
	print('frame_amount sum : ', np.sum(frame_amount))
	mfcc_train_data = mfcc_train_data[:,1:-3].astype(np.float)
	fbank_train_data = fbank_train_data[:,1:-3].astype(np.float)
	#mfcc_train_data[mfcc_train_data[:,0].argsort()]


	# Read labels #
	label = []
	with open(argv[0] + 'label/train.lab') as myfile:
		for line in myfile:
			input_buffer = line.strip().split(',')
			id_value = input_buffer[0].split('_')
			#input_buffer[0] = r.match(id_value[1]).group(2)
			#input_buffer[0] = id_value[1]
			input_buffer[1] = np.where(first_map[:,0] == input_buffer[1])[0][0]
			input_buffer = input_buffer + id_value
			label.append(input_buffer)
		print('success')
	#new = [list(convert(sublist)) for sublist in label]
	#label = np.array(new,dtype = object)
	#label = np.array(label).astype(np.float)
	label = np.array(label, dtype = object)
	#print(label[0])
	label = label[label[:,-3].argsort(kind='mergesort')]
	label = label[label[:,-2].argsort(kind='mergesort')]
	print(label.shape)
	#print(label[0:100])

	lb = preprocessing.LabelBinarizer()
	new_label = label[:,1].astype(np.int)
	lb.fit(new_label)
	#print('lb.classes_: ', lb.classes_)
	one_hot_label = lb.transform(new_label)
	print(one_hot_label[0])
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
	x_f, y_f, y_one_hot_f = batch_padding(fbank_train_data, label[:,1], one_hot_label, frame_amount, MAX_FRAME_LENGTH, 69)
	x_m, y_m, y_one_hot_m = batch_padding(mfcc_train_data, label[:,1], one_hot_label, frame_amount, MAX_FRAME_LENGTH, 39)

	filepath1 = 'my_model-fbank-cnn.hdf5'
	model1 = load_model(filepath1)
	prediction1 = model1.predict(x_f, batch_size = 128, verbose=1)
	#print('prediction.shape : ', prediction1.shape)
	#outfile = TemporaryFile()
	np.save('prediction1.npy', prediction1)	
	np.save('one_hot_f.npy', y_one_hot_f)
	filepath2 = 'my_model-2119.hdf5'
	model2 = load_model(filepath2)
	prediction2 = model2.predict(x_m, batch_size = 128, verbose=1)
	print('prediction.shape : ', prediction2.shape)
	np.save('prediction2.npy', prediction2)	
	'''
	prediction = []
	for i in range(prediction1.shape[0]):
		for j in range(prediction2.shape[1]):
			prediction.append(np.append(prediction1[i,j,:], prediction2[i,j,:]))
	print('len(prediction) :', len(prediction))
	print("Data preprocessing took %.2f seconds" % (timestamp2 - timestamp1))
	print('start random forest')
	RFC = RandomForestClassifier(n_estimators=100,verbose=1,max_features=30,n_jobs=-1,random_state=20171027)
	RFC.fit(prediction, y_one_hot_f.reshape(-1, 48))
	joblib.dump(RFC, 'RFC.pkl')
	'''
#model = load_model('my_model.hdf5')

if __name__ == '__main__':
	main(sys.argv[1:])																																	