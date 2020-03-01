import os, sys, glob

def get_news(n):
	trainN=300; testN=100
	video = '19_news/'+str(n);name = str(n)+'_bmold_win3';start = 0;
	print(video,name)

	rootdir = os.path.join(os.getcwd(),'../Deep3DFaceReconstruction/output/render/')
	srcdir = os.path.join(rootdir,video)
	srcdir2 = srcdir.replace(video,video+'/bm')

	if 'bmold' not in name:
		cmd = "cd "+rootdir+"/..; matlab -nojvm -nosplash -nodesktop -nodisplay -r \"alpha_blend_news('" + video + "'," + str(start) + "," + str(trainN+testN) + "); quit;\""
	else:
		cmd = "cd "+rootdir+"/..; matlab -nojvm -nosplash -nodesktop -nodisplay -r \"alpha_blend_newsold('" + video + "'," + str(start) + "," + str(trainN+testN) + "); quit;\""
	os.system(cmd)
	if not os.path.exists('datasets/list/trainA'):
		os.makedirs('datasets/list/trainA')
	if not os.path.exists('datasets/list/trainB'):
		os.makedirs('datasets/list/trainB')
	f1 = open('datasets/list/trainA/%s.txt'%name,'w')
	f2 = open('datasets/list/trainB/%s.txt'%name,'w')
	if 'win3' in name:
		start1 = start + 2
	else:
		start1 = start
	for i in range(start1,start+trainN):
		if 'bmold' not in name:
			print(os.path.join(srcdir2,'frame%d_render_bm.png'%i),file=f1)
		else:
			print(os.path.join(srcdir2,'frame%d_renderold_bm.png'%i),file=f1)
		print(os.path.join(srcdir,'frame%d.png'%i),file=f2)
	f1.close()
	f2.close()
	if not os.path.exists('datasets/list/testA'):
		os.makedirs('datasets/list/testA')
	if not os.path.exists('datasets/list/testB'):
		os.makedirs('datasets/list/testB')
	f1 = open('datasets/list/testA/%s.txt'%name,'w')
	f2 = open('datasets/list/testB/%s.txt'%name,'w')
	for i in range(start+trainN,start+trainN+testN):
		if 'bmold' not in name:
			print(os.path.join(srcdir2,'frame%d_render_bm.png'%i),file=f1)
		else:
			print(os.path.join(srcdir2,'frame%d_renderold_bm.png'%i),file=f1)
		print(os.path.join(srcdir,'frame%d.png'%i),file=f2)
	f1.close()
	f2.close()

def save_each_60(folder):
	pths = sorted(glob.glob(folder+'/*.pth'))
	for pth in pths:
		epoch = os.path.basename(pth).split('_')[0]
		if epoch == '60':
			continue
		os.remove(pth)

n = int(sys.argv[1])
gpu_id = int(sys.argv[2])

# prepare training data, and write two txt as training list
get_news(n)

# prepare arcface feature
cmd = 'cd arcface/; python test_batch.py --imglist trainB/%d_bmold_win3.txt --gpu %d' % (n,gpu_id)
os.system(cmd)
cmd = 'cd arcface/; python test_batch.py --imglist testB/%d_bmold_win3.txt --gpu %d' % (n,gpu_id)
os.system(cmd)


# fine tune the mapping
n = str(n)
cmd = 'python train.py --dataroot %s_bmold_win3 --name memory_seq_p2p/%s --model memory_seq --continue_train --epoch 0 --epoch_count 1 --lambda_mask 2 --lr 0.0001 --display_env memory_seq_%s --gpu_ids %d --niter 60 --niter_decay 0' % (n,n,n,gpu_id)
os.system(cmd)
save_each_60('checkpoints/memory_seq_p2p/%s'%n)

epoch = 60
cmd = 'python test.py --dataroot %s_bmold_win3 --name memory_seq_p2p/%s --model memory_seq --num_test 200 --epoch %d --gpu_ids %d --imagefolder images%d' % (n,n,epoch,gpu_id,epoch)
os.system(cmd)