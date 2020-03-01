#coding:utf-8
import librosa
import python_speech_features
import numpy as np
import os
import glob
import torch
import cv2
import sys

def get_mfcc(video, srcdir, tardir):
	test_file = os.path.join(srcdir,video)
	save_file = os.path.join(tardir,video[:-4]+'.npy')
	if os.path.exists(save_file):
		mfcc = np.load(save_file)
		return mfcc
	speech, sr = librosa.load(test_file, sr=16000)
	mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)
	np.save(save_file, mfcc)
	return mfcc

if __name__ == '__main__':
	# FOR LRW, all videos
	srcdir = '/home4/yiran/Dataset/LRW/lipread_mp4/'
	tardir = 'mfcc/lrw/'
	print(len(sorted(glob.glob(srcdir+'/*'))))
	for w in sorted(glob.glob(srcdir+'/*')):
		word = os.path.basename(w)
		# for train
		if not os.path.exists(os.path.join(tardir,word,'train')):
			os.makedirs(os.path.join(tardir,word,'train'))
		#print(word,len(sorted(glob.glob(os.path.join(srcdir,word,'train','*.mp4')))))
		for v in sorted(glob.glob(os.path.join(srcdir,word,'train','*.mp4'))):
			video = '%s/train/%s'%(word,os.path.join(os.path.basename(v)))
			get_mfcc(video, srcdir, tardir)
		# for test
		if not os.path.exists(os.path.join(tardir,word,'test')):
			os.makedirs(os.path.join(tardir,word,'test'))
		#print(word,len(sorted(glob.glob(os.path.join(srcdir,word,'test','*.mp4')))))
		for v in sorted(glob.glob(os.path.join(srcdir,word,'test','*.mp4'))):
			video = '%s/test/%s'%(word,os.path.join(os.path.basename(v)))
			get_mfcc(video, srcdir, tardir)
