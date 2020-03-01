import os, sys
import glob
import numpy as np
import pdb
from PIL import Image
import cv2


def merge_with_bigbg(audiobasen,n):
	start=0;ganepoch=60;audioepoch=99
	seamlessclone = 1
	person = str(n)
	if os.path.exists(os.path.join('../audio/',audiobasen+'.wav')):
		in_file = os.path.join('../audio/',audiobasen+'.wav')
	elif os.path.exists(os.path.join('../audio/',audiobasen+'.mp3')):
		in_file = os.path.join('../audio/',audiobasen+'.mp3')
	else:
		print('audio file not exists, please put in %s'%os.path.join(os.getcwd(),'../audio'))
		return
	
	audio_exp_name = 'atcnet_pose0_con3/'+person
	audiomodel=os.path.join(audio_exp_name,audiobasen+'_%d'%audioepoch)
	sample_dir = os.path.join('../results/',audiomodel)
	ganmodel='memory_seq_p2p/%s'%person;post='_full9'
	seq='rseq_'+person+'_'+audiobasen+post
	if audioepoch == 49:
		seq='rseq_'+person+'_'+audiobasen+'_%d%s'%(audioepoch,post)

	coeff_dir = os.path.join(sample_dir,'reassign')

	sample_dir2 = '../../render-to-video/results/%s/test_%d/images%s/'%(ganmodel,ganepoch,seq)
	os.system('cp '+sample_dir2+'/R_'+person+'_reassign2-00002_blend2_fake.png '+sample_dir2+'/R_'+person+'_reassign2-00000_blend2_fake.png')
	os.system('cp '+sample_dir2+'/R_'+person+'_reassign2-00002_blend2_fake.png '+sample_dir2+'/R_'+person+'_reassign2-00001_blend2_fake.png')

	video_name = os.path.join(sample_dir,'%s_%swav_results%s.mp4'%(person,audiobasen,post))
	command = 'ffmpeg -loglevel panic -framerate 25  -i ' + sample_dir2 +  '/R_' + person + '_reassign2-%05d_blend2_fake.png -c:v libx264 -y -vf format=yuv420p ' + video_name
	os.system(command)
	command = 'ffmpeg -loglevel panic -i ' + video_name + ' -i ' + in_file + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
	os.system(command)
	os.remove(video_name)
	
	if not os.path.exists(os.path.join('../../Data',str(n),'transbig.npy')):
		cmd = 'cd ../../Deep3DFaceReconstruction/; python demo_preprocess.py %d %d' % (n,n+1)
		os.system(cmd)
	transdata = np.load(os.path.join('../../Data',str(n),'transbig.npy'))
	w2 = transdata[0]
	h2 = transdata[1]
	t0 = transdata[2]
	t1 = transdata[3]

	coeffs = glob.glob(coeff_dir+'/*.npy')
	transbigbgdir = os.path.join(sample_dir,'trans_bigbg')
	if not os.path.exists(transbigbgdir):
		os.mkdir(transbigbgdir)
	for i in range(len(coeffs)):
		data = np.load(coeff_dir+'/%05d.npy'%i)
		assigni = data[-1]
		if seamlessclone == 0:
			# direct paste
			img = Image.open('../../Data/'+person+'/frame%d.png'%assigni)
			img1 = Image.open(sample_dir2+'/R_'+person+'_reassign2-%05d_blend2_fake.png'%i)
			img1 = img1.resize((w2,h2),resample = Image.LANCZOS)
			img.paste(img1,(t0,t1,t0+img1.size[0],t1+img1.size[1]))
			img.save(os.path.join(transbigbgdir,'%05d.png'%i))
		else:
			# seamless clone
			img = cv2.imread('../../Data/'+person+'/frame%d.png'%assigni)
			img1 = cv2.imread(sample_dir2+'/R_'+person+'_reassign2-%05d_blend2_fake.png'%i)
			img1 = cv2.resize(img1,(w2,h2),interpolation=cv2.INTER_LANCZOS4)
			mask = np.ones(img1.shape,img1.dtype) * 255
			center = (t0+int(img1.shape[0]/2),t1+int(img1.shape[1]/2))
			output = cv2.seamlessClone(img1,img,mask,center,cv2.NORMAL_CLONE)
			cv2.imwrite(os.path.join(transbigbgdir,'%05d.png'%i),output)
	
	transbigbgdir = os.path.join(sample_dir,'trans_bigbg')
	video_name = os.path.join(sample_dir,'%s_%swav_results_transbigbg.mp4'%(person,audiobasen))
	command = 'ffmpeg -loglevel panic -framerate 25  -i ' + transbigbgdir +  '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + video_name
	os.system(command)
	command = 'ffmpeg -loglevel panic -i ' + video_name + ' -i ' + in_file + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
	os.system(command)
	os.remove(video_name)
	print('saved to', video_name.replace('.mp4','.mov'))

audiobasen=sys.argv[1]
n = int(sys.argv[2])

if __name__ == "__main__":
	merge_with_bigbg(audiobasen,n)
