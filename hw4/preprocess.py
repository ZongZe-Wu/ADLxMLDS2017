import numpy as np
import sys
import re
import pickle
import json
import codecs
import unicodedata
import collections
from scipy.ndimage import imread
import os
import time
from skimage.transform import resize
import random

def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f, encoding="bytes")

def wrong_image(od_image, od_preprocess_features, inv_map, image_order):
	timestamp1 = time.time()
	wrong_od_image = []
	for key, value in od_preprocess_features.items():
		while(True):
			inv_key = random.choice(list(inv_map.keys()))
			#print(value, ' ', inv_key)
			a = re.split(' ',inv_key)
			b = re.split(' ', value)
			if a[0] != b[0] and a[2] != b[2]:
				print(value, ' ', inv_key)
				break
		#break
		wrong_od_image.append(od_image[image_order.index(random.choice(inv_map[inv_key]))])
	print(np.array(wrong_od_image).shape)
	np.save('training/wrong_od_image.npy', np.array(wrong_od_image))

def Sent2Vec(tags):
	timestamp1 = time.time()
	feature_dict = np.load('feature_dict.npy', encoding='bytes').item()
	#feature_dict = np.load('feature_dict.npy').item()
	#feature_dict = load_obj('feature_dict')
	#feature_dict = {}
	#print(type(feature_dict))
	#vec_features = {}
	new_feature_dict = {}
	for key, value in feature_dict.items():
		key = key.decode("utf-8")
		new_feature_dict[key] = value
	#for key, value in tags.items():
		#vec_features[key] = new_feature_dict[value]
	vec_features = [[key] + new_feature_dict[tags[key]] for key, value in tags.items()]
	print(np.array(vec_features).shape)
	timestamp2 = time.time()
	print("%.2f seconds for reading text and transfer into skip-thought vector" % (timestamp2 - timestamp1))
	return vec_features
def wrong_tags(tags):
	timestamp1 = time.time()
	color_hair = ['orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair',
	'blue hair', 'black hair', 'brown hair', 'blonde hair']
	color_eyes = ['gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
	'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
	len_hair = len(color_hair)
	len_eyes = len(color_eyes)
	timestamp1 = time.time()
	feature_dict = np.load('feature_dict.npy', encoding='bytes').item()
	new_feature_dict = {}
	for key, value in feature_dict.items():
		key = key.decode("utf-8")
		new_feature_dict[key] = value
	wrong_tags_vec = [[key] + new_feature_dict[color_hair[random.choice(list(range(0,value[0]))+list(range(value[0]+1,len_hair)))]+' '+ \
					color_eyes[random.choice(list(range(0,value[1]))+list(range(value[1]+1,len_eyes)))]] for key, value in tags.items()]
	print(np.array(wrong_tags_vec).shape)
	timestamp2 = time.time()
	print("%.2f seconds for reading text and transfer into wrong_tags_vec vector" % (timestamp2 - timestamp1))
	return wrong_tags_vec
def main(argv):
	color_hair = ['orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair',
	'blue hair', 'black hair', 'brown hair', 'blonde hair', 'unknown hair']
	color_eyes = ['gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
	'green eyes', 'brown eyes', 'red eyes', 'blue eyes', 'unknown eyes']
	possible_hair = ['blonde hair', 'long hair', 'purple hair', 'red hair', 'short hair', 'blue hair', 'pink hair',
	'green hair', 'white hair', 'gray hair', 'black hair', 'brown hair', 'aqua hair', 'orange hair', 'pubic hair', 'damage hair', 'michairu', 'unknown hair']
	possible_eyes = ['pink eyes', 'purple eyes', 'aqua eyes', 'bicolored eyes', 'black eyes', 'blue eyes', 'yellow eyes', 'green eyes',
	'orange eyes', 'red eyes', 'brown eyes', 'gray eyes', '11 eyes', 'eyes rutherford', 'unknown eyes']
	#possible_hair = []
	#possible_eyes = []
	image_id = {}
	with open(argv[0] + 'tags_clean.csv') as tags_file:
		for line in tags_file:
			#print(line)
			content_dict = collections.OrderedDict()
			input_buffer = [i.strip() for i in re.split(',|:|\t',line.strip())]
			for i in range(1, len(input_buffer)-1, 2):
				content_dict[input_buffer[i]] = input_buffer[i+1]
				'''
				if 'hair' in input_buffer[i]:
					if input_buffer[i] not in possible_hair:
						possible_hair.append(input_buffer[i])
				if 'eyes' in input_buffer[i]:
					if input_buffer[i] not in possible_eyes:
						possible_eyes.append(input_buffer[i])
				'''
			image_id[input_buffer[0]] = content_dict
	print(image_id['2054'])
	print('possible_hair : ', possible_hair)
	print('===============================')
	print('possible_eyes : ', possible_eyes)
	print("length of image_id : %d " % len(image_id))
	preprocess_features = {}
	preprocess_tags = {}
	new_tag ={}
	for key, value in image_id.items():
		hair_tag = ''
		eyes_tag = ''
		for key_tags, value_1 in value.items():
			if key_tags in color_hair:
				hair_tag = key_tags
			if key_tags in color_eyes:
				eyes_tag = key_tags
		if hair_tag == '':
			hair_tag = 'unknown hair'
		if eyes_tag == '':
			eyes_tag = 'unknown eyes'
		if hair_tag + ' ' + eyes_tag != 'unknown hair unknown eyes':
			buf = [0]*13
			buf1 = [0]*12
			preprocess_features[int(key)] = hair_tag + ' ' + eyes_tag
			preprocess_tags[int(key)] = (color_hair.index(hair_tag), color_eyes.index(eyes_tag))
			buf[color_hair.index(hair_tag)]=1
			buf1[color_eyes.index(eyes_tag)] = 1
			new_tag[int(key)] = buf+buf1
	
	#print(preprocess_features[0])
	od_preprocess_features = collections.OrderedDict(sorted(preprocess_features.items()))
	od_preprocess_label = collections.OrderedDict(sorted(preprocess_tags.items()))
	od_new_tag = collections.OrderedDict(sorted(new_tag.items()))
	np.save('training/od_new_tag.npy', od_new_tag)
	od_tag = []
	inv_tag={}
	'''
	for k, v in od_new_tag.items():
		if v not in inv_tag:
			inv_tag[v]=[k]
		else:
			inv_tag[v].append(k)
		od_tag.append(v)
	'''
	np.save('training/od_tag.npy', od_tag)
	print(np.array(od_tag).shape)
	print(len(inv_tag))
	np.save('training/od_preprocess_features.npy', od_preprocess_features)
	#print(od_preprocess_features)
	#inv_map = {v: k for k, v in od_preprocess_features.items()}
	inv_map = {}
	arr_features = []
	image_order = []
	arr_num_label = []
	for k, v in od_preprocess_features.items():
		arr_features.append(v)
		image_order.append(k)
		if v not in inv_map:
			inv_map[v] = [k]
		else:
			inv_map[v].append(k)
	print(type(inv_map))
	for k, v in od_preprocess_label.items():
		arr_num_label.append(v)
	#print(arr_num_label)
	np.save('training/image_order.npy', image_order)
	#inv_map = collections.OrderedDict(sorted(inv_map.items()))
	np.save('training/inv_map.npy', inv_map)
	print(len(inv_map))
	print(inv_map['aqua hair brown eyes'])
	total_len = 0
	file = open('training_size2.csv','w')
	for key, value in inv_map.items():
		#print(key, ':', len(value), end='\t')
		file.write(key)
		file.write('\t')
		file.write(str(len(value)))
		file.write('\n')
		total_len += len(value)
	file.close()
	print('len of all data : ', total_len)
	training_image_list = []
	directory = argv[0] + 'faces/'
	'''
	for filename in os.listdir(directory):
		if filename.endswith(".jpg"):
			training_image_list.append(imread(os.path.join(directory, filename)))
			filename_list.append(filename.replace('.jpg',''))
		else:
			continue
	'''
	'''
	timestamp1 = time.time()
	for k,v in od_preprocess_features.items():
		filename = str(k) + '.jpg'
		img = imread(os.path.join(directory, filename))
		training_image_list.append(resize(img, (64,64), mode='constant'))
	print(np.array(training_image_list).shape)
	np.save('training/image_training.npy', np.array(training_image_list))
	timestamp2 = time.time()
	print("%.2f seconds for reading image" % (timestamp2 - timestamp1))
	'''
	
	return od_preprocess_features, od_preprocess_label, inv_map, arr_features, image_order, arr_num_label
if __name__ == '__main__':
	char_tags, char_label, inv_map, arr_features, image_order, arr_num_label = main(sys.argv[1:])
	np.save('training/num_label.npy', arr_num_label)
	#np.save('training_tag_str.npy', arr_features)
	#training_image = np.load('training/image_training.npy')
	#print(training_image.shape)
	#wrong_image(training_image, char_tags, inv_map, image_order)
	#wrong_tags_feature = wrong_tags(char_label)
	#np.save('training/wrong_text_training.npy', wrong_tags_feature)
	#preprocess_feature_vectors = Sent2Vec(char_tags)
	#np.save('training/text_training.npy', preprocess_feature_vectors)