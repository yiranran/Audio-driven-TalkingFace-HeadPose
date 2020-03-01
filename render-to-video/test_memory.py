"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
#encoding:utf-8
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pdb


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    opt.netG = 'unetac_adain_256'
    opt.model = 'test'
    opt.Nw = 3
    opt.norm = 'batch'
    opt.dataset_mode = 'single_multi'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch), refresh=0, folder=opt.imagefolder)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    N = dataset.__len__()
    features = torch.zeros((N,1,1,512)).cuda(opt.gpu_ids[0])
    control = 1
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        feature = model.forward_getfeat()
        feature5 = model.forward_getfeatk(5)
        features[i] = feature
    if control == 4:
        poly = PolynomialFeatures(degree=512)
        fea=features.cpu().numpy()
        for m in range(0,512):
            x = np.arange(1, features.shape[0]+1, 1)
            y = fea[:,0,0,m]
            z1 = np.polyfit(x, y, 10)
            p1 = np.poly1d(z1)
            yvals=p1(x) 
            fea[:,0,0,m]=yvals
        features=torch.Tensor(fea).cuda(opt.gpu_ids[0])

    #np.save('features.npy',features.cpu().numpy())
    for i, data in enumerate(dataset):
        model.set_input(data)
        if control == 0:
            feature = features[0]
        elif control == 1:
            # interpolation
            M = 25
            if i % M == 0 or i == N-1:
                feature = features[i]
            else:
                l = i // M * M
                r = min(l + M, N-1)
                feature = (i-l)/float(r-l) * (features[r]-features[l]) + features[l]
        elif control == 2:
            # average by 3
            if i == 0:
                feature = features[i]
            elif i == 1:
                feature = torch.mean(features[i-1:i+1],dim=0)
            else:
                feature = torch.mean(features[i-2:i+1],dim=0)
        elif control == 3:
            # average by all
            feature = torch.mean(features,dim=0)
        elif control == 4:
            # fit
            feature = features[i]
        model.forward_withfeat(feature)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
    print('control',control)
