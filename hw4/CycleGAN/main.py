import numpy as np
import os
import random
from skimage.transform import resize
from scipy.ndimage import imread
from scipy.misc import imsave
import tensorflow as tf
import losses, model
import argparse
import time
slim = tf.contrib.slim

class CycleGAN(object):
	def __init__(self, pool_size, lambda_a, lambda_b, base_lr, max_step, skip):
		
		self._pool_size = pool_size
		self._lambda_a = lambda_a
		self._lambda_b = lambda_b
		self._num_imgs_to_save = 20
		self._base_lr = base_lr
		self._max_step = max_step
		self._skip = skip

		# self.fake_images_A = np.zeros((self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_CHANNELS))
		# self.fake_images_B = np.zeros((self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_CHANNELS))

		self.fake_images_A = np.zeros((self._pool_size, 1, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS))
		self.fake_images_B = np.zeros((self._pool_size, 1, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS))

	def model_setup(self):
		"""
		This function sets up the model to train.
		self.input_A/self.input_B -> Set of training images.
		self.fake_A/self.fake_B -> Generated images by corresponding generator
		of input_A and input_B
		self.lr -> Learning rate variable
		self.cyc_A/ self.cyc_B -> Images generated after feeding
		self.fake_A/self.fake_B to corresponding generator.
		This is use to calculate cyclic loss
		"""

		self.input_a = tf.placeholder(tf.float32, [1, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS], name='input_a')
		self.input_b = tf.placeholder(tf.float32, [1, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS], name='input_b')
		self.fake_pool_A = tf.placeholder(tf.float32, [None, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS], name='fake_pool_A')
		self.fake_pool_B = tf.placeholder(tf.float32, [None, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS], name='fake_pool_B')


		self.global_step = slim.get_or_create_global_step()

		self.num_fake_inputs = 0

		self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

		inputs = {
			'images_a': self.input_a,
			'images_b': self.input_b,
			'fake_pool_a': self.fake_pool_A,
			'fake_pool_b': self.fake_pool_B,
		}

		outputs = model.get_outputs(
			inputs, skip=self._skip)

		self.prob_real_a_is_real = outputs['prob_real_a_is_real']
		self.prob_real_b_is_real = outputs['prob_real_b_is_real']
		self.fake_images_a = outputs['fake_images_a']
		self.fake_images_b = outputs['fake_images_b']
		self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
		self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

		self.cycle_images_a = outputs['cycle_images_a']
		self.cycle_images_b = outputs['cycle_images_b']

		self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
		self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

	def compute_losses(self):
		"""
		In this function we are defining the variables for loss calculations
		and training model.
		d_loss_A/d_loss_B -> loss for discriminator A/B
		g_loss_A/g_loss_B -> loss for generator A/B
		*_trainer -> Various trainer for above loss functions
		*_summ -> Summary variables for above loss functions
		"""
		# loss for auto-encoder
		cycle_consistency_loss_a = \
			self._lambda_a * losses.cycle_consistency_loss(
				real_images=self.input_a, generated_images=self.cycle_images_a,
			)
		cycle_consistency_loss_b = \
			self._lambda_b * losses.cycle_consistency_loss(
				real_images=self.input_b, generated_images=self.cycle_images_b,
			)
		# lsgan loss
		lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
		lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)
		# total loss for generator
		g_loss_A = \
			cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
		g_loss_B = \
			cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a
		# total loss for discriminator
		d_loss_A = losses.lsgan_loss_discriminator(
			prob_real_is_real=self.prob_real_a_is_real,
			prob_fake_is_real=self.prob_fake_pool_a_is_real,
		)
		d_loss_B = losses.lsgan_loss_discriminator(
			prob_real_is_real=self.prob_real_b_is_real,
			prob_fake_is_real=self.prob_fake_pool_b_is_real,
		)

		optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

		self.model_vars = tf.trainable_variables()

		d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
		g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
		d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
		g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

		self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
		self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
		self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
		self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

		for var in self.model_vars:
			print(var.name)

		# Summary variables for tensorboard
		self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
		self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
		self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
		self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

	# def save_images(self, sess, epoch):
	# 	"""
	# 	Saves input and output images.
	# 	:param sess: The session.
	# 	:param epoch: Currnt epoch.
	# 	"""
	# 	if not os.path.exists(self._images_dir):
	# 		os.makedirs(self._images_dir)

	# 	names = ['inputA_', 'inputB_', 'fakeA_',
	# 			 'fakeB_', 'cycA_', 'cycB_']

	# 	with open(os.path.join(
	# 			self._output_dir, 'epoch_' + str(epoch) + '.html'
	# 	), 'w') as v_html:
	# 		for i in range(0, self._num_imgs_to_save):
	# 			print("Saving image {}/{}".format(i, self._num_imgs_to_save))
	# 			inputs = sess.run(self.inputs)
	# 			fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
	# 				self.fake_images_a,
	# 				self.fake_images_b,
	# 				self.cycle_images_a,
	# 				self.cycle_images_b
	# 			], feed_dict={
	# 				self.input_a: inputs['images_i'],
	# 				self.input_b: inputs['images_j']
	# 			})

	# 			tensors = [inputs['images_i'], inputs['images_j'],
	# 					   fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

	# 			for name, tensor in zip(names, tensors):
	# 				image_name = name + str(epoch) + "_" + str(i) + ".jpg"
	# 				imsave(os.path.join(self._images_dir, image_name),
	# 					   ((tensor[0] + 1) * 127.5).astype(np.uint8)
	# 					   )
	# 				v_html.write(
	# 					"<img src=\"" +
	# 					os.path.join('imgs', image_name) + "\">"
	# 				)
	# 			v_html.write("<br>")

	def fake_image_pool(self, num_fakes, fake, fake_pool):
		"""
		This function saves the generated image to corresponding
		pool of images.
		It keeps on feeling the pool till it is full and then randomly
		selects an already stored image and replace it with new one.
		"""
		if num_fakes < self._pool_size:
			fake_pool[num_fakes] = fake
			return fake
		else:
			p = random.random()
			if p > 0.5:
				random_id = random.randint(0, self._pool_size - 1)
				temp = fake_pool[random_id]
				fake_pool[random_id] = fake
				return temp
			else:
				return fake
	def load_data(self):
		dir_a = '/home/zong-ze/Downloads/sample_face_3w5'
		dir_b = '/home/zong-ze/MLDS/hw4/image_training.npy'
		self.faces_image = []
		filename_list = []
		timestamp1 = time.time()
		'''
		for filename in os.listdir(dir_a):
			if filename.endswith(".jpg"):
				img=imread(os.path.join(dir_a, filename))
				img=resize(img,(64,64),mode='constant')
				self.faces_image.append(img)
				filename_list.append(filename.replace('.jpg',''))
			else:
				continue
		self.faces_image = np.array(self.faces_image)
		np.save('face_image.npy', self.faces_image)
		np.save('file_name.npy', filename_list)
		'''
		self.faces_image = np.load('face_image.npy')
		timestamp2 = time.time()
		print("%.2f seconds for reading celebA image" % (timestamp2 - timestamp1))
		timestamp1 = time.time()
		self.anima_image = np.load(dir_b)
		timestamp2 = time.time()
		print("%.2f seconds for reading anima image" % (timestamp2 - timestamp1))
		self.amount1 = int(self.faces_image.shape[0])
		self.amount2 = int(self.anima_image.shape[0])

	def batch_gen(self, batch_size):
		self.index1 = self._index%self.amount1
		self.index2 = self._index%self.amount2
		bg_image_real_face = self.faces_image[self.index1:self.index1+batch_size]
		bg_image_anima = self.anima_image[self.index2:self.index2+batch_size]
		self._index += batch_size

		return bg_image_real_face, bg_image_anima	

	def train(self):
		"""Training Function."""
		# Load Dataset from the dataset folder
		# self.inputs = data_loader.load_data(
		# 	self._dataset_name, self._size_before_crop,
		# 	True, self._do_flipping)

		# Build the network
		self.model_setup()

		# Loss function calculations
		self.compute_losses()

		# Initializing the global variables
		init = (tf.global_variables_initializer(),
				tf.local_variables_initializer())
		saver = tf.train.Saver()

		# max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]
		max_images = 33431		
		self._index = 0
		self.load_data()
		with tf.Session() as sess:
			sess.run(init)

			# Restore the model to run the model from last checkpoint
			# if self._to_restore:
			# 	chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
			saver.restore(sess, 'Model1/model.ckpt')

			writer = tf.summary.FileWriter('logs/')

			# if not os.path.exists(self._output_dir):
			# 	os.makedirs(self._output_dir)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			# Training Loop
			for epoch in range(sess.run(self.global_step), self._max_step):
				print("In the epoch ", epoch)
				saver.save(sess, 'Model2/model.ckpt')

				# Dealing with the learning rate as per the epoch number
				if epoch < 100:
					curr_lr = self._base_lr
				else:
					curr_lr = self._base_lr - \
						self._base_lr * (epoch - 100) / 100

				for i in range(0, max_images):
					print("In the epoch ", epoch, end='\t')
					print("Processing batch {}/{}".format(i, max_images))
					images_real_face, images_anima = self.batch_gen(1)
					#inputs = sess.run(self.inputs)

					# Optimizing the G_A network
					_, fake_B_temp, summary_str, cycle_images_a = sess.run(
						[self.g_A_trainer,
						 self.fake_images_b,
						 self.g_A_loss_summ,
						 self.cycle_images_a],
						feed_dict={
							self.input_a:
								images_real_face,
							self.input_b:
								images_anima,
							self.learning_rate: curr_lr
						}
					)
					writer.add_summary(summary_str, epoch * max_images + i)

					fake_B_temp1 = self.fake_image_pool(
						self.num_fake_inputs, fake_B_temp, self.fake_images_B)

					# Optimizing the D_B network
					_, summary_str = sess.run(
						[self.d_B_trainer, self.d_B_loss_summ],
						feed_dict={
							self.input_a:
								images_real_face,
							self.input_b:
								images_anima,
							self.learning_rate: curr_lr,
							self.fake_pool_B: fake_B_temp1
						}
					)
					writer.add_summary(summary_str, epoch * max_images + i)

					# Optimizing the G_B network
					_, fake_A_temp, summary_str, cycle_images_b = sess.run(
						[self.g_B_trainer,
						 self.fake_images_a,
						 self.g_B_loss_summ,
						 self.cycle_images_b],
						feed_dict={
							self.input_a:
								images_real_face,
							self.input_b:
								images_anima,
							self.learning_rate: curr_lr
						}
					)
					writer.add_summary(summary_str, epoch * max_images + i)

					fake_A_temp1 = self.fake_image_pool(
						self.num_fake_inputs, fake_A_temp, self.fake_images_A)

					# Optimizing the D_A network
					_, summary_str = sess.run(
						[self.d_A_trainer, self.d_A_loss_summ],
						feed_dict={
							self.input_a:
								images_real_face,
							self.input_b:
								images_anima,
							self.learning_rate: curr_lr,
							self.fake_pool_A: fake_A_temp1
						}
					)
					writer.add_summary(summary_str, epoch * max_images + i)

					writer.flush()
					self.num_fake_inputs += 1
					if (i+1) % 100 == 0:
						#cycle_images_a = current_generator(fake_images_b, 'g_B', skip=skip)
						#cycle_images_b = current_generator(fake_images_a, 'g_A', skip=skip)
						imsave('fake_anima/'+'ep_'+ str(i) + '.jpg', fake_B_temp[0])
						imsave('fake_real/'+'ep_'+ str(i) + '.jpg', fake_A_temp[0])
						imsave('cycle_real2anima/'+'ep_'+ str(i) + '.jpg', cycle_images_b[0])
						imsave('cycle_anima2real/'+'ep_'+ str(i) + '.jpg', cycle_images_a[0])
						self.little_test(sess)
				sess.run(tf.assign(self.global_step, epoch + 1))

			coord.request_stop()
			coord.join(threads)
			writer.add_graph(sess.graph)


	def little_test(self,sess):
		dir_a = '/home/zong-ze/'
		dir_a = '/home/zong-ze/Downloads/sample_face_3w5'
		for i in range(2):
			#filename = 'test'+str(i)+'.jpg'
			for filename in os.listdir(dir_a):
				if filename.endswith(".jpg"):
					img=imread(os.path.join(dir_a, filename))
					img=img.reshape(1,64,64,3)
					break
			fake_B_temp = sess.run(
					self.fake_images_b
				, feed_dict={
					self.input_a: img
				})
			imsave('test_'+ str(i) + '.jpg', fake_B_temp[0])

	def test(self):
		"""Test Function."""
		print("Testing the results")

		# self.inputs = data_loader.load_data(
		# 	self._dataset_name, self._size_before_crop,
		# 	False, self._do_flipping)

		self.model_setup()
		#self.compute_losses()
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)

			#chkpt_fname = tf.train.latest_checkpoint('Model1/')
			saver.restore(sess, 'Model2/model.ckpt')

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			# self._num_imgs_to_save = cyclegan_datasets.DATASET_TO_SIZES[
			# 	self._dataset_name]
			# self.save_images(sess, 0)
			'''
			for i in range(0, self._num_imgs_to_save):
				print("Saving image {}/{}".format(i, self._num_imgs_to_save))
				inputs = sess.run(self.inputs)
				fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
					self.fake_images_a,
					self.fake_images_b,
					self.cycle_images_a,
					self.cycle_images_b
				], feed_dict={
					self.input_a: inputs['images_i'],
					self.input_b: inputs['images_j']
				})
			'''
			#dir_a = '/home/zong-ze/'
			dir_a = ''
			#dir_a = '/home/zong-ze/Downloads/sample_face_3w5'
			for i in range(0,6):
				filename = 'test'+str(i)+'.jpg'

				img=imread(os.path.join(dir_a, filename))
				img=resize(img,(64,64),mode='constant')
				img=img.reshape(1,64,64,3)

				fake_B_temp = sess.run(
						self.fake_images_b
					, feed_dict={
						self.input_a: img
					})
				imsave('testing/'+'test_'+ str(i) + '.jpg', fake_B_temp[0])
			coord.request_stop()
			coord.join(threads)


# @click.command()
# @click.option('--to_train',
# 			  type=click.INT,
# 			  default=True,
# 			  help='Whether it is train or false.')
# @click.option('--log_dir',
# 			  type=click.STRING,
# 			  default=None,
# 			  help='Where the data is logged to.')
# @click.option('--config_filename',
# 			  type=click.STRING,
# 			  default='train',
# 			  help='The name of the configuration file.')
# @click.option('--checkpoint_dir',
# 			  type=click.STRING,
# 			  default='',
# 			  help='The name of the train/test split.')
# @click.option('--skip',
# 			  type=click.BOOL,
# 			  default=False,
# 			  help='Whether to add skip connection between input and output.')
def main():
	"""
	:param to_train: Specify whether it is training or testing. 1: training; 2:
	 resuming from latest checkpoint; 0: testing.
	:param log_dir: The root dir to save checkpoints and imgs. The actual dir
	is the root dir appended by the folder with the name timestamp.
	:param config_filename: The configuration file.
	:param checkpoint_dir: The directory that saves the latest checkpoint. It
	only takes effect when to_train == 2.
	:param skip: A boolean indicating whether to add skip connection between
	input and output.
	"""
	parser = argparse.ArgumentParser('')
	parser.add_argument('--train', action='store_true', help='whether train GAN')
	parser.add_argument('--test', action='store_true', help='whether test GAN')
	args = parser.parse_args()

	# if not os.path.isdir(log_dir):
	# 	os.makedirs(log_dir)

	# with open(config_filename) as config_file:
	# 	config = json.load(config_file)

	lambda_a = 10.0
	lambda_b = 10.0
	pool_size =  50

	base_lr = 0.0002
	max_step = 25
	skip = True
	cyclegan_model = CycleGAN(pool_size, lambda_a, lambda_b,
							  base_lr, max_step, skip)

	if args.train:
		cyclegan_model.train()
	else:
		cyclegan_model.test()


if __name__ == '__main__':
	main()