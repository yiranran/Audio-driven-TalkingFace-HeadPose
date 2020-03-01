#encoding:utf-8
import os
import glob
import shutil
import numpy as np
from scipy.io import loadmat,savemat
import cv2
import pdb
import sys
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def IOU(a,b):#quyu > 0
	I = np.sum((a*b)>0)#a>0 and b>0
	U = np.sum((a+b)>0)#a>0 or b>0
	return I/U

def smooth(x,window_len=11,window='hanning'):
	if x.ndim != 1:
		raise(ValueError, "smooth only accepts 1 dimension arrays.")
	if x.size < window_len:
		raise(ValueError, "Input vector needs to be bigger than window size.")
	if window_len<3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	if window == 'flat':
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')

	y=np.convolve(w/w.sum(),s,mode='valid')
	return y[int(window_len/2):-int(window_len/2)]

def nearest(sucai, query):
	diff = np.abs(np.tile(query, [sucai.shape[0],1]) - sucai)
	cost = np.sum(diff[:,:3],axis=1) #+ 0.1 * np.sum(diff[:,3:],axis=1)
	I = np.argmin(cost)
	#print(query, diff[I,:], cost[I])
	#print(query,sucai[I])
	return I

def nearestIoU(sucai2, query):
	cost = np.zeros(sucai2.shape[0])
	for i in range(sucai2.shape[0]):
		cost[i] = IOU(query, sucai2[i])
	I = np.argmax(cost)
	#print(cost,cost[I])
	#pdb.set_trace()
	return I

def nearest2(sucai, query, sucai2, lastI, lamda=1./255., choice = 1):
	diff1 = np.abs(np.tile(query, [sucai.shape[0],1]) - sucai)
	cost1 = np.sum(diff1[:,:3],axis=1)
	##
	#cost1[lastI] = 100
	lbg = sucai2[lastI]
	if choice == 1:
		## BG L1 similarity
		diff2 = np.abs(sucai2 - np.tile(lbg, [sucai.shape[0],1,1,1]))
		cost2 = np.mean(diff2, axis=(1,2,3))
		#pdb.set_trace()
		I = np.argmin(cost1+cost2*lamda)
	elif choice == 2:
		## BG IOU
		cost2 = np.zeros(cost1.shape)
		for i in range(len(sucai2)):
			cost2[i] = IOU(sucai2[i], lbg) #iou larger better
		I = np.argmin(cost1+(1-cost2)*lamda)
		#pdb.set_trace()
	elif choice == 0:
		# coeff similarity
		lbg = sucai[lastI]
		diff2 = np.abs(np.tile(lbg, [sucai.shape[0],1]) - sucai)
		cost2 = np.sum(diff2[:,:3],axis=1)
		#pdb.set_trace()
		I = np.argmin(cost1+cost2*lamda)
	#print(query,sucai[I],cost1[I],np.sum(diff1[I,:3]))
	if I != lastI:
		print(I)
	return I

def nearest2IoU(query, sucai2, lastI, lamda=1./255.):
	cost1 = np.zeros(sucai2.shape[0])
	for i in range(sucai2.shape[0]):
		cost1[i] = IOU(query, sucai2[i])
	##
	#cost1[lastI] = -100
	lbg = sucai2[lastI]
	## BG IOU
	cost2 = np.zeros(cost1.shape)
	for i in range(len(sucai2)):
		cost2[i] = IOU(sucai2[i], lbg) #iou larger better
	I = np.argmax(cost1+cost2*lamda)
	#pdb.set_trace()
	return I

def choose_bg_gexinghua2_reassign2(video, audio, start, audiomodel='', num=300, debug=0, tran=0, speed=2, aaxis=2):
	print('choose_bg_gexinghua2',video,audio,start,audiomodel)
	rootdir = '../../Deep3DFaceReconstruction/'
	matdir = os.path.join(rootdir,'output/coeff',video)
	pngdir = os.path.join(rootdir,'output/render',video)
	L = 64
	if audiomodel == '':
		folder_to_process = '../results/atcnet_pose0/' + audio
	else:
		folder_to_process = '../results/' + audiomodel
	files = sorted(glob.glob(os.path.join(folder_to_process,'*.npy')))
	tardir = os.path.join('../results/chosenbg','%s_%s'%(audio,video))
	if audiomodel != '':
		tardir = os.path.join('../results/chosenbg','%s_%s_%s'%(audio,video,audiomodel.replace('/','_')))
	tardir2 = os.path.join(tardir, 'reassign')
	print(tardir2)
	if not os.path.exists(tardir2):
		os.makedirs(tardir2)

	sucai = np.zeros((num,6))
	lm_5p = np.zeros((num,2))
	for i in range(start,start+num):
		coeff = loadmat(os.path.join(matdir,'frame%d.mat')%i)
		sucai[i-start,:3] = coeff['coeff'][:,224:227]
		sucai[i-start,3:] = coeff['coeff'][:,254:257]
		if tran:
			lm_5p[i-start,:] = np.mean(coeff['lm_5p'],axis=0)
	
	cnt = 0
	Is = []
	period = 0
	periods = []
	# find max and mins
	N = len(files)
	datas = np.zeros((N,3)) #存姿势的3个角度
	datasall = np.zeros((N,70)) #存表情和姿势的所有系数
	for i in range(N):
		temp = np.load(files[i])
		datas[i] = temp[L:L+3]
		datasall[i] = temp
	#pdb.set_trace()

	# 得到关键帧Ids
	#y = smooth(datas[:,2],window_len=7)
	y = [0,0,0]
	Ids = [0,0,0]
	y0 = [0,0,0]
	n = 0
	if aaxis in [0,1,2]:
		axises = [aaxis]
		thre = N/10.
	elif aaxis in [5]:
		axises = [2]
		thre = N/10.
	else:
		axises = [0,1,2]
		thre = N/5.
	print(aaxis, axises)
	for k in axises:
		y[k] = smooth(datas[:,k],window_len=7)
		y0[k] = y[k]
		#if debug:
			#plt.plot(y0[k])
			#plt.plot(datas[:,k])
		#plt.legend(['axis0','axis1','axis2'])
		# local maxima
		maxIds = argrelextrema(y[k],np.greater)
		# local minima
		minIds = argrelextrema(y[k],np.less)
		Ids[k] = np.concatenate((maxIds[0],minIds[0]))
		Ids[k] = np.sort(Ids[k])
		n += Ids[k].shape[0]
		y[k] = y0[k][Ids[k]]

	while n > thre:
		n = 0
		for k in axises:
			maxIds = argrelextrema(y[k],np.greater,order=2)
			minIds = argrelextrema(y[k],np.less,order=2)
			Ids[k] = np.concatenate((Ids[k][maxIds],Ids[k][minIds]))
			Ids[k] = np.sort(Ids[k])
			n += Ids[k].shape[0]
			y[k] = y0[k][Ids[k]]

	# 关键帧: 0, Ids, N-1
	# 画图看选的关键帧在哪里
	if debug:
		pdb.set_trace()
		#plt.plot(0,datas[0,2],'+')
		#for k in axises:
		#	for i in range(len(Ids[k])):
		#		plt.plot(Ids[k][i],y0[k][Ids[k][i]],'+')
		#plt.plot(N-1,datas[N-1,2],'+')
		
		#plt.savefig('theta.jpg')
	if aaxis == 4:
		Ids = np.concatenate((Ids[0],Ids[1],Ids[2]))
	elif aaxis == 5:
		Ids = Ids[2]
	else:
		Ids = Ids[aaxis]
	Ids = np.sort(np.unique(Ids))
	print(Ids)
	for i in range(1,Ids.shape[0]):
		if Ids[i] - Ids[i-1] < 3:
			#print(Ids[i-1],Ids[i], datas[Ids[i-1],:], datas[Ids[i],:])
			if np.max(np.abs(datas[Ids[i],:])) > np.max(np.abs(datas[Ids[i-1],:])):
				Ids[i-1] = -1
			else:
				Ids[i] = -1
	Ids = np.delete(Ids,np.argwhere(Ids==-1))
	print(Ids.shape[0],N)
	print(Ids)
	

	# 查找和关键帧姿势最接近背景
	if debug:
		tempdir = os.path.join(tardir, 'temp')
		if not os.path.exists(tempdir):
			os.makedirs(tempdir)
	Ids=np.insert(Ids,0,0)
	Ids=np.append(Ids,N-1)
	Is=np.zeros(Ids.shape)
	I = nearest(sucai[:,:3], datas[0])
	Is[0] = I
	for i in range(1,Ids.shape[0]):
		period = Ids[i] - Ids[i-1]
		#sucait = sucai[max(0,I-2*period):min(num,I+2*period),:3]
		sucait = sucai[max(0,int(I-speed*period)):min(num,I+int(speed*period)),:3]
		In = nearest(sucait, datas[Ids[i]])
		#I = max(0,I-2*period) + In
		I = max(0,int(I-speed*period)) + In
		print(Ids[i],I, Ids[i-1])
		Is[i] = I
		if debug:
			if tran == 0:
				for j in range(Ids[i-1], Ids[i]+1):
					shutil.copy(os.path.join(pngdir,'frame%d.png'%(I+start)),
						os.path.join(tempdir,'%05d.png'%j))
			else:
				for j in range(Ids[i-1], Ids[i]+1):
					shutil.copy(os.path.join(pngdir,'frame%d_input2.png'%(I+start)),
						os.path.join(tempdir,'%05d.png'%j))
	
	if debug:
		os.system('ffmpeg -loglevel panic -framerate 25 -i ' + tempdir + '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + tempdir + '.mp4')

	print(Ids,Is)
	# reassign,重新设置姿势的系数
	assigns = [0] * N
	startI = 0
	for i in range(Ids.shape[0]-1):
		l = Ids[i+1] - Ids[i]
		assigns[Ids[i]] = int(Is[i])
		for j in range(1,l):
			assigns[Ids[i]+j] = int(round(float(j)/l*(Is[i+1]-Is[i]) + Is[i]))
		startI += l
	assigns[Ids[-1]] = int(Is[-1])
	print(assigns)
	if not os.path.exists(folder_to_process+'/reassign'):
		os.mkdir(folder_to_process+'/reassign')
	for i in range(N):
		if tran == 0:
			data = datasall[i]
			if aaxis == 5 and i in Ids:
				#pdb.set_trace()
				data[L+3:L+6] = sucai[assigns[i],3:6]
				continue
			data[L:L+6] = sucai[assigns[i]]
		else:
			# 把trans存在.npy里
			data = np.zeros((L+9))
			data[:L] = datasall[i,:L]
			data[L:L+6] = sucai[assigns[i]]
			data[L+6:L+8] = lm_5p[assigns[i]]
			data[L+8] = assigns[i]+start
			#print(i,'assigni',assigns[i]+start,'lm_5p',lm_5p[assigns[i]])
		savename = os.path.join(folder_to_process,'reassign','%05d.npy'%i)
		np.save(savename, data)
		if tran == 0 or tran == 2:
			shutil.copy(os.path.join(pngdir,'frame%d.png'%(assigns[i]+start)),
				os.path.join(tardir2,'%05d.png'%i))
		elif tran == 1:
			#print(os.path.join(pngdir,'frame%d_input2.png'%(assigns[i]+start)))
			shutil.copy(os.path.join(pngdir,'frame%d_input2.png'%(assigns[i]+start)),
				os.path.join(tardir2,'%05d.png'%i))
	
	if debug:
		os.system('ffmpeg -loglevel panic -framerate 25 -i ' + tardir2 + '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + tardir2 + '.mp4')
	
	return tardir2

