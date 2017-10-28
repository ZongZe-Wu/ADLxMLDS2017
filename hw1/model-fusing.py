import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Masking, Dropout, Bidirectional, GRU
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def main(argv):


	prediction1 = np.load('prediction1.npy')
	prediction2 = np.load('prediction2.npy')
	one_hot_label = np.load('one_hot_f.npy')
	frame_amount = np.load('frame_amount.npy')
	print('prediction1.shape', prediction1.shape)
	print('prediction2.shape', prediction2.shape)
	print('one_hot_label.shape', one_hot_label.shape)
	print('frame_amount.shape', frame_amount.shape)


	prediction = []
	one_hot_label_reshape = []
	for i in range(prediction1.shape[0]):
		for j in range(1000):
			prediction.append(np.append(prediction1[i,j,:], prediction2[i,j,:]))
			one_hot_label_reshape.append(one_hot_label[i,j,:])
	prediction = np.array(prediction)
	one_hot_label_reshape = np.array(one_hot_label_reshape)
	print('prediction.shape : ', prediction.shape)
	#one_hot_label = one_hot_label.reshape(-1,48)
	print('one_hot_label_reshape.shape', one_hot_label_reshape.shape)

	#RFC = RandomForestClassifier(n_estimators=25,verbose=1,max_features=40,n_jobs=-1,random_state=20171027)
	#RFC.fit(prediction, one_hot_label)
	#joblib.dump(RFC, 'RFC.pkl')
	filepath = argv[0] + '.hdf5'

	model = Sequential()
	model.add(Dense(256, input_shape = (96,), activation = 'relu'))
	model.add(Dense(256, activation = 'relu'))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(48, activation = 'softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
	model_check = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
	# Train the model, iterating on the data in batches of 32 samples
	reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,patience=5, min_lr=0.001, mode = 'max', verbose=1)
	model.fit(prediction, one_hot_label_reshape, epochs=120, batch_size=128, callbacks=[model_check, reduce_lr], validation_split = 0.25)

if __name__ == '__main__':
	main(sys.argv[1:])