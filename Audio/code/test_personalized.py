#encoding:utf-8
#test different audio
import os
from choose_bg_gexinghua2_reassign import choose_bg_gexinghua2_reassign2
from trans_with_bigbg import merge_with_bigbg
import glob
import pdb
from PIL import Image
import numpy as np
import sys

def getsingle(srcdir,name,varybg=0,multi=0):
	srcroot = os.getcwd()
	if not varybg:
		imgs = glob.glob(os.path.join(srcroot,srcdir,'*_blend.png'))
		print('srcdir',os.path.join(srcroot,srcdir,'*_blend.png'))
	else:
		imgs = glob.glob(os.path.join(srcroot,srcdir,'*_blend2.png'))
		print('srcdir',os.path.join(srcroot,srcdir,'*_blend2.png'))
	if not os.path.exists('../../render-to-video/datasets/list/testSingle'):
		os.makedirs('../../render-to-video/datasets/list/testSingle')
	f1 = open('../../render-to-video/datasets/list/testSingle/%s.txt'%name,'w')
	imgs = sorted(imgs)
	if multi:
		imgs = imgs[2:]
	for im in imgs:
		print(im, file=f1)
	f1.close()

gpu_id = 0 if len(sys.argv) < 4 else int(sys.argv[3])
start=0;ganepoch=60;audioepoch=99


audiobasen=sys.argv[1]
n = int(sys.argv[2])#person id

if __name__ == "__main__":
	person = str(n)
	if os.path.exists(os.path.join('../audio/',audiobasen+'.wav')):
		in_file = os.path.join('../audio/',audiobasen+'.wav')
	elif os.path.exists(os.path.join('../audio/',audiobasen+'.mp3')):
		in_file = os.path.join('../audio/',audiobasen+'.mp3')
	else:
		print('audio file not exists, please put in %s'%os.path.join(os.getcwd(),'../audio'))
		exit(-1)

	audio_exp_name = 'atcnet_pose0_con3/'+person
	audiomodel=os.path.join(audio_exp_name,audiobasen+'_%d'%audioepoch)
	sample_dir = os.path.join('../results/',audiomodel)
	ganmodel='memory_seq_p2p/%s'%person;post='_full9'
	pingyi = 1;
	seq='rseq_'+person+'_'+audiobasen+post
	if audioepoch == 49:
		seq='rseq_'+person+'_'+audiobasen+'_%d%s'%(audioepoch,post)


	## 1.audio to 3dmm
	if not os.path.exists(sample_dir+'/00000.npy'):
		add = '--model_name ../model/%s/atcnet_lstm_%d.pth --pose 1 --relativeframe 0' % (audio_exp_name,audioepoch)
		print('python atcnet_test1.py --device_ids %d %s --sample_dir %s --in_file %s' % (gpu_id,add,sample_dir,in_file))
		os.system('python atcnet_test1.py --device_ids %d %s --sample_dir %s --in_file %s' % (gpu_id,add,sample_dir,in_file))

	## 2.background matching
	speed=1
	num = 300
	bgdir = choose_bg_gexinghua2_reassign2('19_news/'+person, audiobasen, start, audiomodel, num=num, tran=pingyi, speed=speed)


	## 3.render to save_dir
	coeff_dir = os.path.join(sample_dir,'reassign')
	rootdir = '../../Deep3DFaceReconstruction/output/coeff/'
	tex2_path = ''
	coef_path1 = rootdir+'19_news/'+person+'/frame%d.mat'%start
	save_dir = os.path.join(sample_dir,'R_%s_reassign2'%person)
	relativeframe = 2
	os.system('CUDA_VISIBLE_DEVICES=%d python render_for_view2.py %s %s %s %d %d %s'%(gpu_id,coeff_dir,coef_path1,save_dir, relativeframe,pingyi,tex2_path))


	## 4.blend rendered with background
	srcdir = save_dir
	#if not os.path.exists(save_dir+'/00000_blend2.png'):
	cmd = "cd ../results; matlab -nojvm -nosplash -nodesktop -nodisplay -r \"alpha_blend_vbg('" + bgdir + "','" + srcdir + "'); quit;\""
	os.system(cmd)

	## 5.gan
	sample_dir2 = '../../render-to-video/results/%s/test_%d/images%s/'%(ganmodel,ganepoch,seq)
	#if not os.path.exists(sample_dir2):
	getsingle(save_dir,seq,1,1)
	os.system('cd ../../render-to-video; python test_memory.py --dataroot %s --name %s --netG unetac_adain_256 --model test --Nw 3 --norm batch --dataset_mode single_multi --use_memory 1 --attention 1 --num_test 10000 --epoch %d --gpu_ids %d --imagefolder images%s'%(seq,ganmodel,ganepoch,gpu_id,seq))


	os.system('cp '+sample_dir2+'/R_'+person+'_reassign2-00002_blend2_fake.png '+sample_dir2+'/R_'+person+'_reassign2-00000_blend2_fake.png')
	os.system('cp '+sample_dir2+'/R_'+person+'_reassign2-00002_blend2_fake.png '+sample_dir2+'/R_'+person+'_reassign2-00001_blend2_fake.png')
	
	video_name = os.path.join(sample_dir,'%s_%swav_results%s.mp4'%(person,audiobasen,post))
	command = 'ffmpeg -loglevel panic -framerate 25  -i ' + sample_dir2 +  '/R_' + person + '_reassign2-%05d_blend2_fake.png -c:v libx264 -y -vf format=yuv420p ' + video_name
	os.system(command)
	command = 'ffmpeg -loglevel panic -i ' + video_name + ' -i ' + in_file + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
	os.system(command)
	os.remove(video_name)
	print('saved to',video_name.replace('.mp4','.mov'))

	merge_with_bigbg(audiobasen,n)
