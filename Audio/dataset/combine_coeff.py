import os
import glob
import pdb
from scipy.io import loadmat,savemat
import numpy as np

srcdir = '/home4/yiran/Dataset/LRW/lipread_mp4/'
tardir = '../../Deep3DFaceReconstruction/output/coeff/lrw/'
coeffdir = 'coeff/lrw/'

#for lrw, gather all frames' coeff together
for w in sorted(glob.glob(srcdir+'/*')):
	word = os.path.basename(w)
	if not os.path.exists(os.path.join(coeffdir,word,'train')):
		os.makedirs(os.path.join(coeffdir,word,'train'))
	if not os.path.exists(os.path.join(coeffdir,word,'test')):
		os.makedirs(os.path.join(coeffdir,word,'test'))
	for v in sorted(glob.glob(os.path.join(srcdir,word,'*/*.mp4'))):
		ss = v.split('/')
		video = '%s/%s/%s'%(word,ss[-2],ss[-1][:-4])
		
		# all videos in lrw have 29 frames
		complete = True
		coeff = np.zeros((29,257),np.float32)
		for i in range(29):
			coffpath = os.path.join(tardir,video,'frame%d.mat'%i)
			if not os.path.exists(coffpath):
				complete = False
				break
			data = loadmat(coffpath)
			coeff[i,:] = data['coeff']
		if not complete:
			continue
		save_file = os.path.join(coeffdir,word,ss[-2],ss[-1][:-4]+'.npy')
		np.save(save_file,coeff)
		print(save_file)

