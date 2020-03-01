import os

n = 31 #person id
gpu = 0
audio = '03Fsi1831'

## finetuning on a target person
cmd1='cd Data/; python extract_frame1.py %d.mp4' % n
os.system(cmd1)

cmd2='cd Deep3DFaceReconstruction/; CUDA_VISIBLE_DEVICES=%d python demo_19news.py ../Data/%d' % (gpu,n)
os.system(cmd2)

cmd3='cd Audio/code; python train_19news_1.py %d %d' % (n,gpu)
os.system(cmd3)

cmd4='cd render-to-video; python train_19news_1.py %d %d' % (n,gpu)
os.system(cmd4)

## test
cmd5='cd Audio/code; python test_personalized.py %s %d %d' % (audio,n,gpu)
os.system(cmd5)