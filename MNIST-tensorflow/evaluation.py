import numpy as np
import time,os,sys
import argparse
import util
import pdb
import matplotlib.pyplot as plt

print(util.toYellow("======================================================="))
print(util.toYellow("evaluation.py (evaluating on MNIST)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data,graph,warp,util
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=True)

tf.reset_default_graph()
# build graph
with tf.device("/gpu:0"):
	# ------ define input data ------
	opt.batchSize = 1
	image = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W])
	label = tf.placeholder(tf.int64,shape=[opt.batchSize])
	# ------ generate perturbation ------
	pInit = data.genPerturbations(opt)
	pInitMtrx = warp.vec2mtrx(opt,pInit)
	# ------ build network ------
	image = tf.expand_dims(image,axis=-1)
	imagePert = warp.transformImage(opt,image,pInitMtrx)
	if opt.netType=="CNN":
		output = graph.fullCNN(opt,imagePert)
	elif opt.netType=="STN":
		imageWarpAll = graph.STN(opt,imagePert)
		imageWarp = imageWarpAll[-1]
		output = graph.CNN(opt,imageWarp)
	elif opt.netType=="IC-STN":
		imageWarpAll = graph.ICSTN(opt,image,pInit)
		imageWarp = imageWarpAll[-1]
		output = graph.CNN(opt,imageWarp)
	softmax = tf.nn.softmax(output)
	labelOnehot = tf.one_hot(label,opt.labelN)
	prediction = tf.equal(tf.argmax(softmax,1),label)

# load data
print(util.toMagenta("loading MNIST dataset..."))
trainData,validData,testData = data.loadMNIST("data/MNIST.npz")

# prepare model saver/summary writer
saver = tf.train.Saver(max_to_keep=20)

print(util.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	util.restoreModel(opt,sess,saver,opt.toIt)

	for idx in range(0,len(validData["image"])):

		singleImage = validData["image"][idx]
		singleImage = np.reshape(singleImage, [1, 28, 28, 1])
		feed_dict = {image: singleImage}
		imageValue, imageWrapValue = sess.run([image, imageWarp], feed_dict=feed_dict)
		# pdb.set_trace()
		imageValue = np.reshape(imageValue, [28,28])
		imageWrapValue = np.reshape(imageWrapValue, [28,28])
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		ax1.imshow(imageValue)
		ax2 = fig.add_subplot(122)
		ax2.imshow(imageWrapValue)
		plt.show()

print(util.toYellow("======= EVALUATION DONE ======="))
