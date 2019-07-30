import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import timeit

vgg_struct = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3', 'conv5_4', 'relu5_4')

Layers = {'conv1_1' : 0, 'conv1_2' : 1, 'pool1' : 2,
	'conv2_1' : 3, 'conv2_2' : 4, 'pool2' : 5,
	'conv3_1' : 6, 'conv3_2' : 7, 'conv3_3' : 8, 'conv3_4' : 9, 'pool3' : 10,
	'conv4_1' : 11, 'conv4_2' : 12, 'conv4_3' : 13, 'conv4_4' : 14, 'pool4' : 15,
	'conv5_1' : 16, 'conv5_2' : 17, 'conv5_3' : 18, 'conv5_4' : 19}

conv_index = np.zeros(20, dtype = int)
	

def transform(X):
	"""
	Transform a image data from 4-D to 3-D.
	"""		
	return X.reshape((X.shape[1], X.shape[2], X.shape[3]))


def load_data(file_name):
	"""
	Input an image.

	file_name: the filename of the image.
	"""	
	pic = mpimg.imread(file_name)

	# make it as a 4-D array to match the form in DNN.
	pic_shape = (1,) + pic.shape
	temp = np.zeros(pic_shape, dtype = int)
	temp[0,:,:,:] = pic	
	return temp


def save_picture(X, file_name):
	"""
	Output an image.

	X : 4-D array, size: (1, H, W, 3).
	file_name: the filename of the image.
	"""		
	# make it as a 3-D array
	scipy.misc.imsave(file_name, transform(X)) 

	
def Gram(F):
	"""
	Compute a Gram matrix of feature maps in one layer.

	F: feature maps of one layer, size: (1, H, W, N).
	"""		
	n_filters = tf.shape(F)[3]

	# reshape as a M * N matrix
	F = tf.reshape(F, [-1, n_filters])
	# compute the Gram matrix
	G = tf.matmul(tf.transpose(F), F)
	size = tf.cast(tf.size(F), dtype = tf.float32)
	# average, size = M * N
	G = G / size
	return G


###########################################################################################
##############       Convolution Layer      ###############################################
###########################################################################################

def conv_layer(input_x, out_channel, name):
	"""
	Convolution layer.

	input_x: The input of the convolution layer, size: (1, H, W, N). 
	out_channel: The number of feature maps. 
	"""
	in_channel = input_x.get_shape()[-1] 
	kernel_shape = 3

	with tf.variable_scope(name):
		w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
		weight = tf.get_variable(name = 'weight', shape = w_shape, trainable = False)
		bias = tf.get_variable(name = 'bias', shape = [out_channel], trainable = False)
		# strides [1, x_movement, y_movement, 1]
		conv_out = tf.nn.conv2d(input_x, weight, strides = [1, 1, 1, 1], padding = "SAME")
		cell_out = tf.nn.relu(conv_out + bias)
		return cell_out


###########################################################################################
##############         Pooling Layer        ###############################################
###########################################################################################

def pooling_layer(input_x, name):
	"""
	Pooling Layer.

	input_x: The input of the pooling layer, size: (1, H, W, N).
	"""
	k_size = 2
	with tf.variable_scope(name):
		# strides [1, k_size, k_size, 1]
		pooling_shape = [1, k_size, k_size, 1]
		cell_out = tf.nn.avg_pool(input_x, strides = pooling_shape, ksize = pooling_shape, padding = "SAME")
		return cell_out


###########################################################################################
##############         Neural Network       ###############################################
###########################################################################################

class Deep_Net(object):	
	def __init__(self, x):    	
		"""
		x: The input image, size: (1, H, W, N).
		"""
		self.x = x
		# build the neural networks
		self.construct()


	def construct(self):
		"""
		Build the neural networks: 16 convolution layers and 4 pooling layers
		"""
		conv_layer_1_1 = conv_layer(self.x, 64, 'conv1_1')
		conv_layer_1_2 = conv_layer(conv_layer_1_1, 64, 'conv1_2')
		pooling_layer_1 = pooling_layer(conv_layer_1_2, 'pool1')

		conv_layer_2_1 = conv_layer(pooling_layer_1, 128, 'conv2_1')
		conv_layer_2_2 = conv_layer(conv_layer_2_1, 128, 'conv2_2')
		pooling_layer_2 = pooling_layer(conv_layer_2_2, 'pool2')

		conv_layer_3_1 = conv_layer(pooling_layer_2, 256, 'conv3_1')
		conv_layer_3_2 = conv_layer(conv_layer_3_1, 256, 'conv3_2')
		conv_layer_3_3 = conv_layer(conv_layer_3_2, 256, 'conv3_3')
		conv_layer_3_4 = conv_layer(conv_layer_3_3, 256, 'conv3_4')
		pooling_layer_3 = pooling_layer(conv_layer_3_4, 'pool3')

		conv_layer_4_1 = conv_layer(pooling_layer_3, 512, 'conv4_1')
		conv_layer_4_2 = conv_layer(conv_layer_4_1, 512, 'conv4_2')
		conv_layer_4_3 = conv_layer(conv_layer_4_2, 512, 'conv4_3')
		conv_layer_4_4 = conv_layer(conv_layer_4_3, 512, 'conv4_4')
		pooling_layer_4 = pooling_layer(conv_layer_4_4, 'pool4')

		conv_layer_5_1 = conv_layer(pooling_layer_4, 512, 'conv5_1')
		conv_layer_5_2 = conv_layer(conv_layer_5_1, 512, 'conv5_2')
		conv_layer_5_3 = conv_layer(conv_layer_5_2, 512, 'conv5_3')
		conv_layer_5_4 = conv_layer(conv_layer_5_3, 512, 'conv5_4')

		self.conv1_1 = conv_layer_1_1
		self.conv1_2 = conv_layer_1_2
		self.pool1 = pooling_layer_1

		self.conv2_1 = conv_layer_2_1
		self.conv2_2 = conv_layer_2_2
		self.pool2 = pooling_layer_2

		self.conv3_1 = conv_layer_3_1
		self.conv3_2 = conv_layer_3_2
		self.conv3_3 = conv_layer_3_3
		self.conv3_4 = conv_layer_3_4
		self.pool3 = pooling_layer_3

		self.conv4_1 = conv_layer_4_1
		self.conv4_2 = conv_layer_4_2
		self.conv4_3 = conv_layer_4_3
		self.conv4_4 = conv_layer_4_4
		self.pool4 = pooling_layer_4

		self.conv5_1 = conv_layer_5_1
		self.conv5_2 = conv_layer_5_2
		self.conv5_3 = conv_layer_5_3
		self.conv5_4 = conv_layer_5_4

		self.Layers = [self.conv1_1, self.conv1_2, self.pool1, 
			self.conv2_1, self.conv2_2, self.pool2, 
			self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4, self.pool3, 
			self.conv4_1, self.conv4_2, self.conv4_3, self.conv4_4, self.pool4, 
			self.conv5_1, self.conv5_2, self.conv5_3, self.conv5_4]


	def content_representation(self, content_layer):
		"""
		return the content representation in one layer.
		"""
		return self.Layers[Layers[content_layer]]	


	def style_representation(self, style_layers):	
		"""
		return the style representation on all layers given in the list of 'style_layers'.
		G: a list of Gram matrices. 
		"""
		G = []
		for l in range(len(style_layers)):
			G.append(Gram(self.Layers[Layers[style_layers[l]]]))
		return G	


	def load_trained_net(self, sess, weights):
		"""
		Load the pre-trained network.

		weights: parameters of the pre-trained network.
		"""		
		for name in weights:
			now = weights[name]
			with tf.variable_scope(name, reuse = True):
				sess.run(tf.get_variable('weight').assign(now[0]))
				sess.run(tf.get_variable('bias').assign(now[1]))


###########################################################################################
##############     Load Pre-trained Network      ##########################################
###########################################################################################

def load_pretrained_net(pretrained_net):
	"""
	Load the pre-trained network from a mat file.

	pretrained_net: the filename of a mat file that contains the parameters of a 
		pre-trained network.
	"""		
	data = scipy.io.loadmat(pretrained_net)
	mean_pixel = data['meta']['normalization'][0][0][0][0][2][0][0]
	now = 0
	for name in vgg_struct:
		if name[:4] == 'conv':
			conv_index[now] = vgg_struct.index(name)
			now += 1
	weights = data['layers'][0]    
	param = dict()
	n_conv = 16 #16
	for i in range(n_conv):
		kernels, bias = weights[conv_index[i]][0][0][2][0]
		bias = bias.reshape(bias.shape[0])    
		param[vgg_struct[conv_index[i]]] = [kernels, bias]

	return mean_pixel, param


###########################################################################################
##############     Content Representation      ############################################
###########################################################################################

def content_feature_map(content_pic, name, weights, content_layer):
	"""
	return the content representation of a given image.

	content_pic: the input image, size: (1, H, W, N).
	weights: parameters of the pre-trained network.
	content_layer: the layer used for the content representation.

	P: feature maps of this layer ('content_layer').
	"""
	with tf.variable_scope(name):
		content = tf.constant(content_pic, dtype = tf.float32)
		model_content = Deep_Net(content)
		feature_map = model_content.content_representation(content_layer)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		with tf.variable_scope(name):
			model_content.load_trained_net(sess, weights)
			P = sess.run(feature_map)
	return P


###########################################################################################
##############     Style Representation      ##############################################
###########################################################################################

def style_representation(style_pic, name, weights, style_layers):
	"""
	return the style representation of a given image.

	style_pic: the input image, size: (1, H, W, N).
	weights: parameters of the pre-trained network.
	style_layers: the layers used for the style representation.

	A: style representation of these layers ('style_layers').
	"""
	with tf.variable_scope(name):
		style = tf.constant(style_pic, dtype = tf.float32)
		model_style = Deep_Net(style)
		style_rep = model_style.style_representation(style_layers)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		with tf.variable_scope(name):
			model_style.load_trained_net(sess, weights)
			A = sess.run(style_rep)
	return A


###########################################################################################
########################     Style Transfer      ##########################################
###########################################################################################

def Style_Transfer(content_file, style_file, vgg_data, output_file, 
		content_layer, style_layers, coef_content, coef_style, 
		learning_rate, max_iter, checkpoint_iter, print_iter):
	"""
	Main function: implement the style transfer algorithm. 

	content_file: filename of input image for content reconstruction.
	style_file: filename of input image for style reconstruction. 
	vgg_data: filename of pre-trained model, should be a mat file, like 'imagenet-vgg-verydeep-19.mat'.
	output_file: the location of the output image. 
	content_layer: the layer used for the content representation, like 'conv1_1'.
	style_layers: the layers used for the style representation, like ['conv1_1', 'conv2_1'].
	coef_content: weighting factor for content reconstruction.
	coef_style: weighting factor for style reconstruction.
	learning_rate: the learning rate for Adam algorithm.
	max_iter: the maximum number of iterations.
	checkpoint_iter: evaluate the loss every 'checkpoint_iter' steps.
	print_iter: output the image every 'print_iter' steps. If it is 0, not print.
	"""

	# load the parameters of pre-trained model
	mean_pixel, weights = load_pretrained_net(vgg_data)
	content_pic = load_data(content_file) - mean_pixel
	style_pic = load_data(style_file) - mean_pixel
	print('Size of the content picture: ', transform(content_pic).shape)
	print('Size of the style picture: ', transform(style_pic).shape)
	#print('Original: ', content_pic.mean(axis = (0,1,2)) + mean_pixel)

	# content representation for the given content image.
	P = content_feature_map(content_pic, 'Network_content', weights, content_layer) 
	# style representation for the given style image.
	A = style_representation(style_pic, 'Network_style', weights, style_layers) 
	L = len(A)
	#print(P.shape, A[L-1].shape, L)

	x = tf.get_variable('mix', shape = content_pic.shape, 
						initializer = tf.random_normal_initializer())    

	# fit a model that mixes the content and style, model_mixed
	with tf.variable_scope('Network_mix'):
		model_mixed = Deep_Net(x)
		F = model_mixed.content_representation(content_layer)
		G = model_mixed.style_representation(style_layers)

	# compute loss
	#print(P.size, A[0].size)
	content_loss = tf.nn.l2_loss(F - P)/P.size
	style_loss = 0
	for l in range(L):
		style_loss += tf.nn.l2_loss(G[l] - A[l])/A[l].size
	style_loss = style_loss / L		
	loss = coef_content * content_loss + coef_style * style_loss

	start = timeit.default_timer()
	iter_total = 0	
	best_loss, best_iter, gap = (1e30, 0, 0)
	# parameters for Adam
	beta1, beta2, epsilon = (0.9, 0.999, 1e-8)	
	cur_model_name = 'transfer_{}'.format(int(timeit.default_timer()))

	# Adam for training
	step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

	with tf.Session() as sess:
		#saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

		with tf.variable_scope('Network_mix'):
			model_mixed.load_trained_net(sess, weights)

		start_iter = timeit.default_timer()
		while (iter_total < max_iter):
			iter_total += 1
			sess.run(step)

			if iter_total % checkpoint_iter == 0:
				content_loss_now, style_loss_now = sess.run([content_loss, style_loss])
				loss_now = coef_content * content_loss_now + coef_style * style_loss_now

				print('Iteration: {}'.format(iter_total))
				stop_iter = timeit.default_timer()				
				print('Time: ', stop_iter - start_iter)
				start_iter = stop_iter
				print('Loss_content: {}'.format(content_loss_now))
				print('Loss_style: {}'.format(style_loss_now))
				print('Loss_total: {}'.format(loss_now))
				print('')

				if loss_now < best_loss:
					best_loss = loss_now
					best_iter = iter_total
					mix_pic = sess.run(x) + mean_pixel
					gap = 0
					#saver.save(sess, 'model/{}'.format(cur_model_name))
				else:
					gap += 1		

				if 	gap * checkpoint_iter > 300: break
				if best_loss < 1e-1: break                   

			if print_iter:		
				if iter_total % print_iter == 0:
					output = sess.run(x) + mean_pixel   
					save_picture(output, output_file + '_iter_' + str(iter_total) + '.jpg')
					#print(iter_total, ': ', output.mean(axis = (0,1,2)))
					#print('')

	stop = timeit.default_timer()
	print('Time: ', stop - start)
	print('Best iter: ', best_iter)
	print('Loss: ', best_loss)
	save_picture(mix_pic, output_file + '.jpg')

	return output_file + '.jpg'


if __name__ == '__main__':
	content_file = 'hoovertowernight.jpg'
	style_file = 'starry_night.jpg' 
	vgg_data = 'imagenet-vgg-verydeep-19.mat'
	output_file = 'output_image/mix'
	content_layer = 'conv4_2'
	style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

	learning_rate = 10
	max_iter, checkpoint_iter, print_iter = (1000, 100, 100)
	coef_content, coef_style = (1e-3, 1)


	output_image = Style_Transfer(content_file, style_file, vgg_data, output_file, 
		content_layer, style_layers, coef_content, coef_style, 
		learning_rate, max_iter, checkpoint_iter, print_iter)


