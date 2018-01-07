import tensorflow as tf 
from layers import *

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 64

# The width of each image.
IMG_WIDTH = 64

# The number of color channels per image.
IMG_CHANNELS = 3

ngf = 32
ndf = 64


def get_outputs(inputs, network='tf', skip=False):
	images_a = inputs['images_a']
	images_b = inputs['images_b']

	fake_pool_a = inputs['fake_pool_a']
	fake_pool_b = inputs['fake_pool_b']

	with tf.variable_scope('Model') as scope:
		current_discriminator = build_discriminaor
		current_generator = build_generator_resnet_5blocks

		prob_real_a_is_real = current_discriminator(images_a, 'd_A')
		prob_real_b_is_real = current_discriminator(images_b, 'd_B')

		fake_images_b = current_generator(images_a, 'g_A', skip=skip)
		fake_images_a = current_generator(images_b, 'g_B', skip=skip)

		scope.reuse_variables()

		prob_fake_a_is_real = current_discriminator(fake_images_a, 'd_A')
		prob_fake_b_is_real = current_discriminator(fake_images_b, 'd_B')

		cycle_images_a = current_generator(fake_images_b, 'g_B', skip=skip)
		cycle_images_b = current_generator(fake_images_a, 'g_A', skip=skip)

		scope.reuse_variables()

		prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, 'd_A')
		prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, 'd_B')

		return {
			'prob_real_a_is_real' : prob_real_a_is_real,
			'prob_real_b_is_real' : prob_real_b_is_real,
			'prob_fake_a_is_real' : prob_fake_a_is_real,
			'prob_fake_b_is_real' : prob_fake_b_is_real,
			'prob_fake_pool_a_is_real' : prob_fake_pool_a_is_real,
			'prob_fake_pool_b_is_real' : prob_fake_pool_b_is_real,
			'cycle_images_a' : cycle_images_a,
			'cycle_images_b' : cycle_images_b,
			'fake_images_a' : fake_images_a,
			'fake_images_b' : fake_images_b,
		}



def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
	
	"""build a single block of resnet.
	:param inputres: inputres
	:param dim: dim
	:param name: name
	:param padding: for tensorflow version use REFLECT
	:return: a single block of resnet.
	"""
	# general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
	# 				padding='VALID', name='conv2d', do_norm=True, do_relu=True,
	# 				relufactor=0)
	with tf.variable_scope(name):
		out_res = tf.pad(inputres, [[0, 0], [1, 1], [1,1], [0,0]], padding)
		out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, padding="VALID", name="c1")
		out_res = tf.pad(out_res, [[0, 0], [1, 1], [1,1], [0,0]], padding)
		out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, padding="VALID", name="c2", do_relu=False)

		return tf.nn.relu(out_res+inputres)


def build_generator_resnet_5blocks(inputgen, name='generator', skip=False):
	# general_deconv2d(inputconv, outputshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
	# 				stddev=0.02, padding='VALID', name='deconv2d', do_norm=True,
	# 				do_relu=True, relufactor=0)
	# print('inputgen.shape : ',inputgen.shape)
	with tf.variable_scope(name):
		f = 7
		ks = 3
		padding = 'REFLECT'

		#pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
		o_c1 = general_conv2d(inputgen, ngf, f, f, 1, 1, 0.02, padding="SAME", name="c1") #64*64*3->64*64*ngf
		# print('o_c1.shape', o_c1.shape)
		o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02, padding="SAME", name="c2") #32,32,ngf*2
		# print('o_c2.shape', o_c2.shape)
		o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02, padding="SAME", name="c3") #16,16,ngf*4
		# print('o_c3.shape', o_c3.shape)
		o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
		o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
		o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
		o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
		o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
		o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
		o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
		o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
		o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)
		# print('o_r5.shape', o_r5.shape)
		o_c4 = general_deconv2d(o_r9, [BATCH_SIZE, 32, 32, ngf*2], ngf*2, ks, ks, 2, 2, 0.02, padding="SAME", name="c4")
		# 32,32, ngf*2
		# print('o_c4.shape', o_c4.shape)
		o_c5 = general_deconv2d(o_c4, [BATCH_SIZE, 64, 64, ngf*2], ngf*2, ks, ks, 2, 2, 0.02, padding="SAME", name="c5")
		# 64,64, ngf*2
		# print('o_c5.shape', o_c5.shape)
		o_c6 = general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1, 0.02, padding="SAME", name="c6", do_norm=False, do_relu=False)
		# print('o_c6.shape', o_c6.shape)
		if skip is True:
			out_gen = tf.nn.sigmoid(inputgen + o_c6, 't1')
		else:
			out_gen = tf.nn.sigmoid(o_c6, 't1')

		return out_gen

def build_discriminaor(inputdis, name='discriminator'):
	# general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
	# 				padding='VALID', name='conv2d', do_norm=True, do_relu=True,
	# 				relufactor=0)
	with tf.variable_scope(name):
		f = 4
		o_c1 = general_conv2d(inputdis, ndf, f, f, 2, 2, 0.02, padding="SAME", name="c1", do_norm=False, relufactor=0.2)
		#32
		o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, padding="SAME", name="c2", do_norm=False, relufactor=0.2)
		#16
		o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, padding="SAME", name="c3", do_norm=False, relufactor=0.2)
		#8
		o_c4 = general_conv2d(o_c3, ndf*8, f, f, 2, 2, 0.02, padding="SAME", name="c4", do_norm=False, relufactor=0.2)
		#4
		o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, padding="SAME", name="c5", do_norm=False, do_relu=False)

		return o_c5

def patch_discriminator(inputdisc, name="discriminator"):
	with tf.variable_scope(name):

		f = 4

		patch_input = tf.random_crop(inputdisc, [1, 30, 30, 3])
		o_c1 = general_conv2d(patch_input, ndf, f, f, 2, 2, 0.02, padding="SAME", name="c1", do_norm=False, relufactor=0.2)
		o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, padding="SAME", name="c2", relufactor=0.2) 
		o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, padding="SAME", name="c3", relufactor=0.2)
		o_c4 = general_conv2d(o_c3, ndf*8, f, f, 2, 2, 0.02, padding="SAME", name="c4", relufactor=0.2)
		o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, padding="SAME", name="c5", do_norm=False, do_relu=False)

		return o_c5








