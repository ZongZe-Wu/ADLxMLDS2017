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
def main(argv):
	test = ['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi', 'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi', 'tJHUH9tpqPg_113_118.avi']
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
	print('feature_arr.shape : ', test_feature_list.shape)
	print('len(filename_list) : ', len(test_file))
	features = (filename_list, feature_arr)
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


	saver = tf.train.Saver()
	with tf.Session() as sess:
		graph = graph
		saver.restore(sess, "model.ckpt")
		prediction = sess.run([model.prediction],feed_dict = {model.xs : testing_npy})
		print(prediction.shape)




if __name__ == '__main__':
	main(sys.argv[1:])	