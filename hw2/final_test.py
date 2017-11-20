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
#from new_try import Model
#from dynamic_try import Model
#from dynamic_try import Testmodel
#from dynamic_try_2 import Model
from final import Model
import os
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
def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)
def main(argv):
	test = ['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi', 'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi', 'tJHUH9tpqPg_113_118.avi']
	test_npy = ['klteYv1Uv9A_27_33.avi.npy', '5YJaS2Eswg0_22_26.avi.npy', 'UbmZAe5u5FI_132_141.avi.npy', 'JntMAcTlOF0_50_70.avi.npy', 'tJHUH9tpqPg_113_118.avi.npy']
	#peer~~`
	directory = argv[0] + 'peer_review/feat/'
	peer_filename_list = []
	peer_feature_list = []
	for filename in os.listdir(directory):
		if filename.endswith(".npy"):
			#print(os.path.join(directory, filename))
			peer_feature_list.append(np.vstack((np.load(os.path.join(directory, filename)), np.zeros((50,4096)) ) ) )
			peer_filename_list.append(filename.replace('.npy',''))
		else:
			continue
	peer_feature_arr = np.array(peer_feature_list)


	#for filename in glob.glob(os.path.join(path, '*.txt')):
	directory = argv[0] + 'training_data/feat/'
	filename_list = []
	feature_list = []
	special = []
	for filename in os.listdir(directory):
		if filename.endswith(".npy"):
			#print(os.path.join(directory, filename))
			feature_list.append(np.vstack((np.load(os.path.join(directory, filename)), np.zeros((50,4096)) ) ) )
			filename_list.append(filename.replace('.npy',''))
			if filename == 'YmXCfQm0_CA_50_57.avi.npy':
				special.append(np.vstack((np.load(os.path.join(directory, filename)), np.zeros((50,4096)) ) ) )
		else:
			continue
	special = np.array(special)
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
			if filename in test_npy:
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
	vectorizer = CountVectorizer(tokenizer=TB().tokenize, min_df = 0.00005)
	vectorizer.fit(list(itertools.chain.from_iterable(caption_all)))
	print(vectorizer.transform(['we']))
	#print(vectorizer.vocabulary_)
	#inv_map = {v: k for k, v in vectorizer.vocabulary_.items()}
	
	inverse = [(value, key) for key, value in vectorizer.vocabulary_.items()]
	max_voc_size = max(inverse)[0]+2
	vectorizer.vocabulary_["unknown"] = max_voc_size -1
	inv_map = {v: k for k, v in vectorizer.vocabulary_.items()}
	print(inverse[0])
	print(max_voc_size)
	print('inv_map 0 :', inv_map[0])
	buf_max = 0
	for i in range(len(label_data)):#len(label_data)
		label_buf = []
		seq_len_buf = []
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
			seq_len_buf.append(arr1.shape[0])
			#print(arr.shape)
			label_buf.append(np.append(arr1, np.array([max_voc_size]*(50-arr.shape[0]))))
		#print(np.array(label_buf).shape)
		label[label_data[i]['id']] = (np.array(label_buf), seq_len_buf)
	print('filename_list[0] : ', filename_list[0])
	print('len(label[filename_list[0]]) : ', label[filename_list[0]][0].shape)
	print(buf_max)
	#print(vectorizer.stop_words_)
	#print(find(vectorizer.transform(label[filename_list[0]][0]))[1])
	print(vectorizer.inverse_transform(vectorizer.transform(['.']))[0])
	#print(vectorizer.inverse_transform(vectorizer.transform(['A'])[0].todense()[0]))
	print(label_buf[-1])
	for j in range(len(label_buf[-1])):
		if label_buf[-1][j] != max_voc_size and label_buf[-1][j] != max_voc_size+1:
			print(inv_map[label_buf[-1][j]], end=' ')
	print('\n')

	#tf.reset_default_graph()
	#with tf.variable_scope(tf.get_variable_scope()):
	model = Model(130, 4096, max_voc_size+1, False)
	
	saver = tf.train.Saver()
	#tf.reset_default_graph()
	with tf.Session() as sess:
		#a = sess.run(model.bs_embedding)
		#print(a)
		#saver = tf.train.Saver()
		#optimistic_restore(sess,  "model_new/model_new.ckpt")
		#graph = graph
		#model = Model(130, 4096, 6087, True)
		saver.restore(sess, "Model/model_dyn_final.ckpt")
		#b = sess.run(model.bs_embedding)
		#print(b)
		#if a == b :
		#	print('Fuckkkkkkkkkkk')
		#tf.reset_default_graph()
		#prediction = sess.run(model.prediction,feed_dict = {model.xs : test_feature_list[:64,:,:], model.batch_size : int((test_feature_list).shape[0])})
		prediction = sess.run(model.prediction,feed_dict = {model.for_training : False,model.xs : test_feature_list, model.batch_size : int((test_feature_list).shape[0]), model.bos : [max_voc_size+1]*int((test_feature_list).shape[0])})
		print(np.array(prediction))
		peer_prediction = sess.run(model.prediction,feed_dict = {model.for_training : False,model.xs : peer_feature_arr, model.batch_size : int((peer_feature_arr).shape[0]), model.bos : [max_voc_size+1]*int((peer_feature_arr).shape[0])})
		#tt = sess.run([model.prediction],feed_dict = {model.xs : test_feature_list, model.batch_size : int((test_feature_list).shape[0])})
	#for i in range(50):
	#	for j in range(50):
	#		print(prediction[i][j],end = ',')
	#	print('\n')
	file = open(argv[1],'w')
	for i in range(len(test_file)):
		file.write(test_file[i])
		file.write(',')
		buf = []
		for j in range(len(prediction[i])):
			#if prediction[i][j] != 6086 and prediction[i][j] != 12 and prediction[i][j] != 6087:
				#if j == 0:			
			if prediction[i][j] < max_voc_size: 	
				if j == 0:
					file.write(inv_map[prediction[i][j]])
					file.write(' ')
					buf.append(inv_map[prediction[i][j]])
				elif inv_map[prediction[i][j]] != buf[-1] and prediction[i][j]!=max_voc_size-1:
					file.write(inv_map[prediction[i][j]])
					buf.append(inv_map[prediction[i][j]])
					file.write(' ')
		file.write('\n')

	file.close()

	file = open(argv[2],'w')
	for i in range(len(peer_filename_list)):
		file.write(peer_filename_list[i])
		file.write(',')
		buf = []
		for j in range(len(peer_prediction[i])):
			#if prediction[i][j] != 6086 and prediction[i][j] != 12 and prediction[i][j] != 6087:
				#if j == 0:			
			if peer_prediction[i][j] < max_voc_size:	
				if j == 0:
					file.write(inv_map[peer_prediction[i][j]])
					file.write(' ')
					buf.append(inv_map[peer_prediction[i][j]])
				elif inv_map[peer_prediction[i][j]] != buf[-1] and peer_prediction[i][j]!=max_voc_size-1::
					file.write(inv_map[peer_prediction[i][j]])
					buf.append(inv_map[peer_prediction[i][j]])
					file.write(' ')
		file.write('\n')

	file.close()


if __name__ == '__main__':
	main(sys.argv[1:])	