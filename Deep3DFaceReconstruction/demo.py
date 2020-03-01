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
	# input and output folder
	save_path = 'output/coeff'	
	save_path2 = 'output/render'
	if image_path[-1] == '/':
		image_path = image_path[:-1]
	name = os.path.basename(image_path)
	print(image_path, name)
	img_list = glob.glob(image_path + '/' + '*.txt')
	img_list = [e[:-4]+'.png' for e in img_list]
	already = glob.glob(save_path + '/' + name + '/*.mat')
	already = [e[len(save_path)+1:-4].replace(name,image_path)+'.png' for e in already]
	ret = list(set(img_list).difference(set(already)))
	img_list = ret
	img_list = sorted(img_list)
	print('img_list len:', len(img_list))
	if not os.path.exists(os.path.join(save_path,name)):
		os.makedirs(os.path.join(save_path,name))
	if not os.path.exists(os.path.join(save_path2,name)):
		os.makedirs(os.path.join(save_path2,name))

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
	with tf.Graph().as_default() as graph:

		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
		graph_def = load_graph('network/FaceReconModel.pb')
		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

		# output coefficients of R-Net (dim = 257) 
		coeff = graph.get_tensor_by_name('resnet/coeff:0')

		faceshaper = tf.placeholder(name = "face_shape_r", shape = [1,35709,3], dtype = tf.float32)
		facenormr = tf.placeholder(name = "face_norm_r", shape = [1,35709,3], dtype = tf.float32)
		facecolor = tf.placeholder(name = "face_color", shape = [1,35709,3], dtype = tf.float32)
		rendered = Render_layer(faceshaper,facenormr,facecolor,facemodel,1)

		rstimg = tf.placeholder(name = 'rstimg', shape = [224,224,4], dtype=tf.uint8)
		encode_png = tf.image.encode_png(rstimg)

		with tf.Session() as sess:
			print('reconstructing...')
			for file in img_list:
				n += 1
				# load images and corresponding 5 facial landmarks
				if '_mtcnn' not in image_path:
					img,lm = load_img(file,file[:-4]+'.txt')
				else:
					img,lm = load_img(file,file[:-4].replace(name,name+'_mtcnn')+'.txt')
				file = file.replace(image_path.replace('_mtcnn',''), name)
				# preprocess input image
				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)
				if n==1:
					transform_firstflame=transform_params
				input_img2,lm_new2 = Preprocess2(img,lm,transform_firstflame)

				coef = sess.run(coeff,feed_dict = {images: input_img})
				
				face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,facemodel)
				final_images = sess.run(rendered, feed_dict={faceshaper: face_shape_r.astype('float32'), facenormr: face_norm_r.astype('float32'), facecolor: face_color.astype('float32')})
				result_image = final_images[0, :, :, :]
				result_image = np.clip(result_image, 0., 1.).copy(order='C')
				result_bytes = sess.run(encode_png,{rstimg: result_image*255.0})
				result_output_path = os.path.join(save_path2,file[:-4]+'_render.png')
				with open(result_output_path, 'wb') as output_file:
					output_file.write(result_bytes)

				# reshape outputs
				input_img = np.squeeze(input_img)
				im = Image.fromarray(input_img[:,:,::-1])
				cropped_output_path = os.path.join(save_path2,file[:-4]+'.png')
				im.save(cropped_output_path)

				input_img2 = np.squeeze(input_img2)
				im = Image.fromarray(input_img2[:,:,::-1])
				cropped_output_path = os.path.join(save_path2,file[:-4]+'_input2.png')
				im.save(cropped_output_path)

				# save output files
				savemat(os.path.join(save_path,file[:-4]+'.mat'),{'coeff':coef,'lm_5p':lm_new2-lm_new})
	t2 = time.time()
	print('Total n:', n, 'Time:', t2-t1)

if __name__ == '__main__':
	demo(sys.argv[1])