import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.io import loadmat,savemat
import sys

from preprocess_img import Preprocess,Preprocess2
from load_data import *
from reconstruct_mesh import Reconstruction
from reconstruct_mesh import Reconstruction_for_render, Render_layer
import pdb
import time

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

def demo(image_path):
	# output folder
	save_dir = 'output/coeff'
	save_coeff_path = save_dir + '/' + image_path
	img_list = glob.glob(image_path + '/*.txt')
	img_list = img_list + glob.glob(image_path + '/*/*.txt')
	img_list = img_list + glob.glob(image_path + '/*/*/*.txt')
	img_list = img_list + glob.glob(image_path + '/*/*/*/*.txt')
	img_list = [e[:-4]+'.jpg' for e in img_list]
	already = glob.glob(save_coeff_path + '/*.mat')
	already = already + glob.glob(save_coeff_path + '/*/*.mat')
	already = already + glob.glob(save_coeff_path + '/*/*/*.mat')
	already = already + glob.glob(save_coeff_path + '/*/*/*/*.mat')
	already = [e[len(save_dir)+1:-4]+'.jpg' for e in already]
	ret = list(set(img_list).difference(set(already)))
	img_list = ret
	img_list = sorted(img_list)
	print('img_list len:', len(img_list))
	if not os.path.exists(os.path.join(save_dir,image_path)):
		os.makedirs(os.path.join(save_dir,image_path))
	for img in img_list:
		if not os.path.exists(os.path.join(save_dir,os.path.dirname(img))):
			os.makedirs(os.path.join(save_dir,os.path.dirname(img)))

	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

	# read face model
	facemodel = BFM()
	# read standard landmarks for preprocessing images
	lm3D = load_lm3d()
	n = 0
	t1 = time.time()

	# build reconstruction model
	#with tf.Graph().as_default() as graph,tf.device('/cpu:0'):
	with tf.Graph().as_default() as graph:

		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
		graph_def = load_graph('network/FaceReconModel.pb')
		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

		# output coefficients of R-Net (dim = 257) 
		coeff = graph.get_tensor_by_name('resnet/coeff:0')

		with tf.Session() as sess:
			print('reconstructing...')
			for file in img_list:
				n += 1
				# load images and corresponding 5 facial landmarks
				img,lm = load_img(file,file[:-4]+'.txt')
				# preprocess input image
				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)

				coef = sess.run(coeff,feed_dict = {images: input_img})

				# save output files
				savemat(os.path.join(save_dir,file[:-4]+'.mat'),{'coeff':coef,'lm_5p':lm_new})
	t2 = time.time()
	print('Total n:', n, 'Time:', t2-t1)

if __name__ == '__main__':
	demo(sys.argv[1])