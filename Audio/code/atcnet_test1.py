
#encoding:utf-8
#测试一个随机的wav
import argparse
import scipy.misc
import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import librosa
import python_speech_features

from models import ATC_net

def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def parse_args():
    parser = argparse.ArgumentParser()
  
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument('-i','--in_file', type=str, default='../audio/test.wav')
    parser.add_argument("--model_name",
                        type=str,
                        default="../model/atcnet/atcnet_lstm_24.pth")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="../results/atcnet/test/")
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--lstm', type=bool, default=True)
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
    parser.add_argument('--para_dim', type=int, default=64)
    parser.add_argument('--index', type=str, default='80,144', help='index ranges')
    parser.add_argument('--pose', type=int, default=0, help='whether predict pose')
    parser.add_argument('--relativeframe', type=int, default=0, help='whether use relative frame value for pose')
   

    return parser.parse_args()
config = parse_args()
str_ids = config.index.split(',')
config.indexes = []
for i in range(int(len(str_ids)/2)):
    start = int(str_ids[2*i])
    end = int(str_ids[2*i+1])
    if end > start:
        config.indexes += range(start, end)
#print('indexes', config.indexes)
print('device', config.device_ids)


def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = 'test'
    if config.lstm == True:
        if config.pose == 0:
            generator = ATC_net(config.para_dim)
        else:
            generator = ATC_net(config.para_dim+6)

    test_file = config.in_file
    speech, sr = librosa.load(test_file, sr=16000)
    mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)
    #print(mfcc.shape)

    state_dict = multi2single(config.model_name, 0)
    generator.load_state_dict(state_dict)
    print('load pretrained [{}]'.format(config.model_name))

    if config.cuda:
        generator = generator.cuda()
    generator.eval()
    
    ind = 3
    with torch.no_grad():
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            # take 280 ms segment
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)
        input_mfcc = input_mfcc.unsqueeze(0)
        print(input_mfcc.shape)
        if config.cuda:
            input_mfcc = Variable(input_mfcc.float()).cuda()
        if config.lstm:
            fake_coeff= generator(input_mfcc)
            fake_coeff = fake_coeff.data.cpu().numpy()
            if not os.path.exists(config.sample_dir):
                os.makedirs(config.sample_dir)
            for jj in range(len(fake_coeff[0])):
                name = "%s/%05d.npy"%(config.sample_dir,jj)
                np.save(name, fake_coeff[0,jj])


test()