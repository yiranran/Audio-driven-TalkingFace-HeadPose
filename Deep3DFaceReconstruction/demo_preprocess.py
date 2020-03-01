import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.io import loadmat,savemat
import sys

from preprocess_img import Preprocess
from load_data import *
from reconstruct_mesh import Reconstruction
from reconstruct_mesh import Reconstruction_for_render, Render_layer
import pdb
import time
import matplotlib.pyplot as plt

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

def demo_19news(n1,n2):
	lm3D = load_lm3d()
	n = 0
	for n in range(n1,n2):
		#print(n)
		start = 0
		file = os.path.join('../Data',str(n),'frame%d.png'%start)
		#print(file)
		if not os.path.exists(file[:-4]+'.txt'):
			continue
		img,lm = load_img(file,file[:-4]+'.txt')
		input_img,lm_new,transform_params = Preprocess(img,lm,lm3D) # lm_new 5x2
		input_img = np.squeeze(input_img)
		img1 = Image.fromarray(input_img[:,:,::-1])

		scale = 0.5 * (lm[0][0]-lm[1][0]) / (lm_new[0][0]-lm_new[1][0]) + 0.5 * (lm[3][0]-lm[4][0]) / (lm_new[3][0]-lm_new[4][0])
		#print(scale)
		trans = np.mean(lm-lm_new*scale, axis=0)
		trans = np.round(trans).astype(np.int32)
		w,h = img1.size
		w2 = int(round(w*scale))
		h2 = int(round(h*scale))
		img1 = img1.resize((w2,h2),resample = Image.LANCZOS)
		img.paste(img1,(trans[0],trans[1],trans[0]+img1.size[0],trans[1]+img1.size[1]))
		np.save(os.path.join('../Data',str(n),'transbig.npy'),np.array([w2,h2,trans[0],trans[1]]))
		#print(os.path.join('../Data',str(n),'transbig.npy'))
		img.save('combine.png')

if __name__ == '__main__':
	demo_19news(int(sys.argv[1]),int(sys.argv[2]))
