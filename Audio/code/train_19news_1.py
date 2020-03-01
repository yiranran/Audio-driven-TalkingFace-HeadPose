import librosa
import python_speech_features
import numpy as np
import os, glob, sys

def get_mfcc_extend(video, srcdir, tardir):
	test_file = os.path.join(srcdir,video)
	save_file = os.path.join(tardir,video[:-4]+'.npy')
	if os.path.exists(save_file):
		mfcc = np.load(save_file)
		return mfcc
	speech, sr = librosa.load(test_file, sr=16000)
	speech = np.insert(speech, 0, np.zeros(1920))
	speech = np.append(speech, np.zeros(1920))
	mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)
	if not os.path.exists(os.path.dirname(save_file)):
		os.makedirs(os.path.dirname(save_file))
	np.save(save_file, mfcc)
	return mfcc

def save_each_100(folder):
	pths = sorted(glob.glob(folder+'/*.pth'))
	for pth in pths:
		epoch = int(os.path.basename(pth).split('_')[-1][:-4])
		if epoch % 100 == 99:
			continue
		#print(epoch)
		os.remove(pth)

n = int(sys.argv[1])
gpu_id = int(sys.argv[2])
# check video
mp4 = '../../Data/%d.mp4'%n
if not os.path.exists(mp4):
    print('target video', mp4, 'not exists')
    exit(-1)
# check 3d recon
rootdir = '../../Deep3DFaceReconstruction/output/coeff/19_news/%d' % n
valid = True
for i in range(300):
    if not os.path.exists(os.path.join(rootdir,'frame%d.mat'%i)):
        print(n,'lack','frame%d.mat'%i)
        valid = False
if not valid:
    print('not all 300 frames are reconstructed successfully')
    exit(-1)
# extract mfcc
srcdir = '../../Data/'
tardir = '../dataset/mfcc/19_news'
video = str(n)+'.mp4'
get_mfcc_extend(video, srcdir, tardir)

# fine tune audio
n = str(n)
if not os.path.exists('../model/atcnet_pose0_con3/%s'%n):
    os.makedirs('../model/atcnet_pose0_con3/%s'%n)
if not os.path.exists('../sample/atcnet_pose0_con3/%s'%n):
    os.makedirs('../sample/atcnet_pose0_con3/%s'%n)
if not os.path.exists('../model/atcnet_pose0_con3/%s/atcnet_lstm_99.pth'%n):
    cmd = 'python atcnet.py --pose 1 --relativeframe 0 --dataset news --newsname 19_news/%s --start 0 --model_dir ../model/atcnet_pose0_con3/%s/ --continue_train 1 --lr 0.0001 --less_constrain 1 --smooth_loss 1 --smooth_loss2 1 --model_name ../model/atcnet_lstm_general.pth --sample_dir ../sample/atcnet_pose0_con3/%s --device_ids %d --max_epochs 100' % (n, n, n, gpu_id)
    print(cmd)
    os.system(cmd)
save_each_100('../model/atcnet_pose0_con3/%s'%n)