import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import time
import copy
import python_speech_features
from scipy.io import loadmat
#import utils
import pdb
#EIGVECS = np.load('../basics/S.npy')
#MS = np.load('../basics/mean_shape.npy')



class LRW_1D_lstm_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16  
        self.lmark_root_path = '../dataset/landmark1d'
        self.pca = torch.FloatTensor(np.load('../basics/U_lrw1.npy')[:,:6] ) 
        self.mean = torch.FloatTensor(np.load('../basics/mean_lrw1.npy'))
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

      
    

    def __getitem__(self, index):
        if self.train=='train':
            lmark_path = os.path.join(self.lmark_root_path , self.train_data[index][0] , self.train_data[index][1],self.train_data[index][2], self.train_data[index][2] + '.npy') 
            mfcc_path = os.path.join('../dataset/mfcc/',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy') 
            
            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca) 
            
            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,8)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]
            example_mfcc = torch.FloatTensor(example_mfcc)
            return example_landmark, example_mfcc, landmark, mfccs
        if self.train=='test':
            lmark_path = os.path.join(self.lmark_root_path , self.test_data[index][0] , self.test_data[index][1],self.test_data[index][2], self.test_data[index][2] + '.npy') 
            mfcc_path = os.path.join('../dataset/mfcc/',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy') 
            
            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca) 
            
            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,8)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]
            example_mfcc = torch.FloatTensor(example_mfcc)
            return example_landmark, example_mfcc, landmark, mfccs
       
    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass

class LRW_1D_lstm_3dmm(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 indexes=range(80,144)):
        self.train = train
        self.num_frames = 16  
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "coeff_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "coeff_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        self.indexes = indexes

    def __getitem__(self, index):
        if self.train=='train':
            mfcc_path = os.path.join('../dataset/mfcc/lrw',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy') 
            coeff_path = os.path.join('../dataset/coeff/lrw',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy') 
            
            mfcc = np.load(mfcc_path)
            coeff = np.load(coeff_path)

            r = random.choice(
                [x for x in range(3,8)])
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            #landmark  =lmark[r+1 : r + 17,:]
            coeff = coeff[r+1:r+17,self.indexes]
            return coeff, mfccs
        if self.train=='test':
            mfcc_path = os.path.join('../dataset/mfcc/lrw',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy') 
            coeff_path = os.path.join('../dataset/coeff/lrw',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy') 
            
            mfcc = np.load(mfcc_path)
            coeff = np.load(coeff_path)

            r = random.choice(
                [x for x in range(3,8)])
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            #landmark  =lmark[r+1 : r + 17,:]
            coeff = coeff[r+1:r+17,self.indexes]
            return coeff, mfccs
       
    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass

class LRW_1D_lstm_3dmm_pose(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 indexes=range(80,144),
                 relativeframe=0):
        self.train = train
        self.num_frames = 16  
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "coeff_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "coeff_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        self.indexes = indexes
        self.relativeframe = relativeframe

    def __getitem__(self, index):
        if self.train=='train':
            mfcc_path = os.path.join('../dataset/mfcc/lrw',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy') 
            coeff_path = os.path.join('../dataset/coeff/lrw',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy') 
            
            mfcc = np.load(mfcc_path)
            coeff = np.load(coeff_path)

            r = random.choice(
                [x for x in range(3,8)])
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            #landmark  =lmark[r+1 : r + 17,:]
            L = len(self.indexes)
            coeffc = np.zeros((16,L+6),dtype=np.float32)
            coeffc[:,:L] = coeff[r+1:r+17,self.indexes]
            coeffc[:,L:L+3] = coeff[r+1:r+17,224:227]
            coeffc[:,L+3:L+6] = coeff[r+1:r+17,254:257]
            coeffc2 = coeffc.copy()
            if self.relativeframe == 1:
                coeffc[0,L:L+6] = 0
                coeffc[1:,L:L+6] = coeffc2[1:,L:L+6] - coeffc2[:-1,L:L+6]
            else:
                coeffc[:,L+5] -= 0.5
            coeffc = torch.FloatTensor(coeffc)
            coeffc2 = torch.FloatTensor(coeffc2)
            return coeffc, mfccs, coeffc2
        if self.train=='test':
            mfcc_path = os.path.join('../dataset/mfcc/lrw',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy') 
            coeff_path = os.path.join('../dataset/coeff/lrw',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy') 
            
            mfcc = np.load(mfcc_path)
            coeff = np.load(coeff_path)

            r = random.choice(
                [x for x in range(3,8)])
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            #landmark  =lmark[r+1 : r + 17,:]
            L = len(self.indexes)
            coeffc = np.zeros((16,L+6),dtype=np.float32)
            coeffc[:,:L] = coeff[r+1:r+17,self.indexes]
            coeffc[:,L:L+3] = coeff[r+1:r+17,224:227]
            coeffc[:,L+3:L+6] = coeff[r+1:r+17,254:257]
            coeffc2 = coeffc.copy()
            if self.relativeframe == 1:
                coeffc[0,L:L+6] = 0
                coeffc[1:,L:L+6] = coeffc2[1:,L:L+6] - coeffc2[:-1,L:L+6]
            else:
                coeffc[:,L+5] -= 0.5
            coeffc = torch.FloatTensor(coeffc)
            coeffc2 = torch.FloatTensor(coeffc2)
            return coeffc, mfccs, coeffc2

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass

class News_1D_lstm_3dmm_pose(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 indexes=range(80,144),
                 relativeframe=0,
                 newsname='Learn_English',
                 start=357,
                 trainN=300,
                 testN=100):
        self.train = train
        self.num_frames = 16  
        self.indexes = indexes
        self.relativeframe = relativeframe
        self.newsname = newsname
        self.start = start
        self.trainN = trainN
        self.testN = testN
        if self.train=='train':
            mfcc_path = os.path.join('../dataset/mfcc/',  self.newsname + '.npy')             
            mfcc = np.load(mfcc_path)
            mfccs = []
            ind = 3
            while ind <= int(mfcc.shape[0]/4) - 4:
                # take 280 ms segment
                t_mfcc = mfcc[(ind - 3)*4: (ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
                ind += 1
            mfccs = torch.stack(mfccs, dim = 0)
            print(mfccs.shape) #(24101, 28, 12) for Ben_Shapiro, (16282, 28, 12) for BBC_Carrie_Lam
            self.mfccs = mfccs[self.start:self.start+self.trainN]
            
            coeff = np.zeros((self.trainN,257),dtype=np.float32)
            for i in range(self.trainN):
                coeff_path = os.path.join('../../Deep3DFaceReconstruction/output/coeff/', self.newsname, 'frame%d.mat'%(self.start+i)) 
                #print(coeff_path)
                data = loadmat(coeff_path)
                coeff[i] = data['coeff']
            L = len(self.indexes)
            coeffc = np.zeros((self.trainN,L+6),dtype=np.float32)
            coeffc[:,:L] = coeff[:,self.indexes]
            coeffc[:,L:L+3] = coeff[:,224:227]
            coeffc[:,L+3:L+6] = coeff[:,254:257]
            coeffc2 = coeffc.copy()
            if self.relativeframe == 1:
                coeffc[0,L:L+6] = 0
                coeffc[1:,L:L+6] = coeffc2[1:,L:L+6] - coeffc2[:-1,L:L+6]
            else:
                coeffc[:,L+5] -= 0.5
            self.coeffc = torch.FloatTensor(coeffc)
            self.coeffc2 = torch.FloatTensor(coeffc2)
            print(self.coeffc.shape)

    def __getitem__(self, index):
        if self.train=='train':
            coeffc = self.coeffc[index:index+16]
            mfccs = self.mfccs[index:index+16]
            coeffc2 = self.coeffc2[index:index+16]
            return coeffc, mfccs, coeffc2

    def __len__(self):
        if self.train=='train':
            return self.trainN-15
        elif self.train=='test':
            return self.testN-15
        else:
            pass

class LRW_1D_single_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16  
        self.lmark_root_path = '../dataset/landmark1d'
        self.audio_root_path = '../dataset/audio'
        self.pca = torch.FloatTensor(np.load('../basics/U_lrw1.npy')[:,:6] ) 
        self.mean = torch.FloatTensor(np.load('../basics/mean_lrw1.npy'))
        
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

     
    

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            lmark_path = os.path.join(self.lmark_root_path , self.train_data[index][0] , self.train_data[index][1],self.train_data[index][2], self.train_data[index][2] + '.npy') 
            mfcc_path = os.path.join('../dataset/mfcc/',  self.train_data[index][0],  self.train_data[index][1],  self.train_data[index][2] + '.npy') 
            
            lmark = np.load(lmark_path)

            lmark = torch.FloatTensor(lmark) * 5.0
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca) 
            
            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,25)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]

            while True:
                current_frame_id = random.choice(
                    [x for x in range(3,25)])
                if current_frame_id != r:
                    break           
            t_mfcc =mfcc[( current_frame_id - 3)*4: (current_frame_id + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc)
            landmark  =lmark[current_frame_id , :]
            example_landmark = torch.FloatTensor(example_landmark)
            example_mfcc = torch.FloatTensor(example_mfcc)
            landmark = torch.FloatTensor(landmark)
            landmark = landmark 
            return example_landmark, example_mfcc, landmark, t_mfcc
        if self.train=='test':
            mfcc_path = os.path.join('../dataset/mfcc/',  self.test_data[index][0],  self.test_data[index][1],  self.test_data[index][2] + '.npy') 
            lmark_path = os.path.join(self.lmark_root_path , self.test_data[index][0] , self.test_data[index][1],self.test_data[index][2], self.test_data[index][2] + '.npy') 
            lmark = np.load(lmark_path)
            mfcc = np.load(mfcc_path)
            example_landmark =lmark[3,:]
            example_mfcc = mfcc[0 : 7 * 4, 1 :]
            r =3
            ind = self.test_data[index][3]

            t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc)
            landmark  =lmark[r+ ind,:]
            # example_audio = torch.FloatTensor(example_audio)
            example_mfcc = torch.FloatTensor(example_mfcc)
            # audio = torch.FloatTensor(audio)
            # mfccs = torch.FloatTensor(mfccs)
            landmark = torch.FloatTensor(landmark)
            # landmark = self.transform(landmark)
            landmark = landmark * 5.0
            example_landmark = torch.FloatTensor(example_landmark).view(1,-1)
            example_landmark = example_landmark - self.mean.expand_as(example_landmark)
            example_landmark = torch.mm(example_landmark,self.pca) 
            
            return example_landmark[0], example_mfcc, landmark, t_mfcc
       
    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass

class LRWdataset1D_single_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

                #load righ img
                image_path = '../dataset/regions/' +  self.train_data[index][0]
                landmark_path = '../dataset/landmark1d/' + self.train_data[index][0][:-8] + '.npy'

                landmark = np.load(landmark_path) * 5.0

                right_landmark = landmark[self.train_data[index][1] - 1]
                right_landmark = torch.FloatTensor(right_landmark.reshape(-1))
                
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)

                r = random.choice(
                    [x for x in range(1,30)])
                example_path =   image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r - 1]
                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)

                return example_img, example_landmark, right_img,right_landmark

        elif self.train =='test':
            # try:
                #load righ img
            image_path = '../dataset/regions/' +  self.test_data[index][0]
            landmark_path = '../dataset/landmark1d/' + self.test_data[index][0][:-8] + '.npy'
            landmark = np.load(landmark_path) * 5.0
            right_landmark = landmark[self.test_data[index][1] - 1]
            
            right_landmark = torch.FloatTensor(right_landmark.reshape(-1))
            
            im = cv2.imread(image_path)
            
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.output_shape)
            im = self.transform(im)
            right_img = torch.FloatTensor(im)
            
            example_path =   '../image/musk1_region.jpg'
            example_landmark = np.load('../image/musk1.npy')
            
            example_landmark = torch.FloatTensor(example_landmark.reshape(-1)) * 5.0

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)

            return example_img, example_landmark, right_img,right_landmark
                
        
class LRWdataset1D_lstm_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

                #load righ img
                image_path = '../dataset/regions/' +  self.train_data[index][0]
                landmark_path = '../dataset/landmark1d/' + self.train_data[index][0][:-8] + '.npy'
                current_frame_id =self.train_data[index][1]
                right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
                for jj in range(16):
                    this_frame = current_frame_id + jj
                    image_path =  '../dataset/regions/' +  self.train_data[index][0][:-7] + '%03d.jpg'%this_frame
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, self.output_shape)
                    im = self.transform(im)
                    right_img[jj,:,:,:] = torch.FloatTensor(im)



                landmark = np.load(landmark_path) * 5.0

                right_landmark = landmark[self.train_data[index][1] - 1 : self.train_data[index][1] + 15  ]
              
                right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))
                

                r = random.choice(
                    [x for x in range(1,30)])
                example_path =   image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r - 1]

                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                # print (right_landmark.size())

                return example_img, example_landmark, right_img,right_landmark
            
            
        elif self.train =='test':
            #load righ img
                image_path = '../dataset/regions/' +  self.test_data[index][0]
                landmark_path = '../dataset/landmark1d/' + self.test_data[index][0][:-8] + '.npy'
                current_frame_id =self.test_data[index][1]
                right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
                for jj in range(16):
                    this_frame = current_frame_id + jj
                    image_path =  '../dataset/regions/' +  self.test_data[index][0][:-7] + '%03d.jpg'%this_frame
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, self.output_shape)
                    im = self.transform(im)
                    right_img[jj,:,:,:] = torch.FloatTensor(im)

                landmark = np.load(landmark_path) * 5.0

                right_landmark = landmark[self.test_data[index][1] - 1 : self.test_data[index][1] + 15  ]
             
                right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))
                r = random.choice(
                    [x for x in range(1,30)])
                r = current_frame_id
                example_path =   image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r - 1]

                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                # print (right_landmark.size())

                return example_img, example_landmark, right_img,right_landmark


    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)


class LRWdataset1D_single(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "new_img_small_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "new_img_small_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            while True:
                # try:
                    #load righ img
                image_path = self.train_data[index][0]
                landmark_path = self.train_data[index][1]
                landmark = np.load(landmark_path)

                right_landmark = landmark[self.train_data[index][2]]
                tp = ( np.dot(right_landmark.reshape(1,6), EIGVECS))[0,:].reshape(68,3)
                tp = tp[:,:-1].reshape(-1)
                right_landmark = torch.FloatTensor(tp)
                im = cv2.imread(image_path)
                if im is None:
                    raise IOError
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)

                r = random.choice(
                        [x for x in range(1,30)])
                example_path = image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[r]


                tp = ( np.dot(example_landmark.reshape(1,6), EIGVECS))[0,:].reshape(68,3)
                tp = tp[:,:-1].reshape(-1)


                example_landmark = torch.FloatTensor(tp)

                example_img = cv2.imread(example_path)
                if example_img is None:
                    raise IOError
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)

                return example_img, example_landmark, right_img,right_landmark
               
        elif self.train =='test':
            while True:
                image_path = self.test_data[index][0]
                landmark_path = self.test_data[index][1]
                landmark = np.load(landmark_path)

                right_landmark = landmark[self.test_data[index][2]]
                right_landmark = torch.FloatTensor((MS + np.dot(right_landmark.reshape(1,6), EIGVECS)).reshape(-1))
                print (right_landmark.shape)
                im = cv2.imread(image_path)
                if im is None:
                    raise IOError
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)
                r = random.choice(
                    [x for x in range(1,30)])
                #load example image
                example_path = image_path[:-8] + '_%03d.jpg'%r
                example_landmark = landmark[self.train_data[r][2]]
                example_landmark = torch.FloatTensor((MS + np.dot(example_landmark.reshape(1,6), EIGVECS)).reshape(-1))

                example_img = cv2.imread(example_path)
                if example_img is None:
                    raise IOError
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                return example_img, example_landmark, right_img,right_landmark
        elif  self.train =='demo':
            landmarks = np.load('/home/lchen63/obama_fake.npy')
            landmarks =np.reshape(landmarks, (landmarks.shape[0], 136))
            while True:
                # try:

                    image_path = self.demo_data[index][0]
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, self.output_shape)
                    im = self.transform(im)

                    right_img = torch.FloatTensor(im)
                    example_path = '/mnt/disk1/dat/lchen63/lrw/demo/musk1_region.jpg'

                    example_landmark = landmarks[0]

                   
                    example_lip = cv2.imread(example_path)

                    example_lip = cv2.cvtColor(example_lip, cv2.COLOR_BGR2RGB)
                    example_lip = cv2.resize(example_lip, self.output_shape)
                    example_lip = self.transform(example_lip)


                    right_landmark = torch.FloatTensor(landmarks[index-1])

                    wrong_landmark = right_landmark
                    return example_lip, example_landmark, right_img,right_landmark, wrong_landmark
        

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)

#############################################################grid 
class GRIDdataset1D_single_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_img_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

                #load righ img
                image_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0], '%05d.jpg'%(self.train_data[index][1] + 1))
                
                landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0] + '_norm_lmarks.npy')

                landmark = np.load(landmark_path)

                right_landmark = landmark[self.train_data[index][1]]
                
                right_landmark = torch.FloatTensor(right_landmark.reshape(-1))
              
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img = torch.FloatTensor(im)

                r = random.choice(
                    [x for x in range(1, 76)])
                example_path =   os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0], '%05d.jpg'%(r))
                example_landmark = landmark[r - 1]

                example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

                example_img = cv2.imread(example_path)
                example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_img = cv2.resize(example_img, self.output_shape)
                example_img = self.transform(example_img)
                return example_img, example_landmark, right_img,right_landmark
            
                


        elif self.train =='test':
            # try:
                #load righ img
            image_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0], '%05d.jpg'%(self.test_data[index][1] + 1))
                
            landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0] + '_norm_lmarks.npy')

            landmark = np.load(landmark_path)

            right_landmark = landmark[self.test_data[index][1]]
            
            right_landmark = torch.FloatTensor(right_landmark.reshape(-1))
          
            im = cv2.imread(image_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.output_shape)
            im = self.transform(im)
            right_img = torch.FloatTensor(im)

            r = random.choice(
                [x for x in range(1, 76)])
            example_path =   os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0], '%05d.jpg'%(r))
            example_landmark = landmark[r - 1]
            

            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)
            return example_img, example_landmark, right_img,right_landmark

        elif  self.train =='demo':

            # try:
                #load righ img
            image_path = '/mnt/ssd0/dat/lchen63/lrw/demo/regions/' +  self.demo_data[index][0]
            landmark_path = '/mnt/ssd0/dat/lchen63/lrw/demo/landmark1d/' + self.demo_data[index][1].replace('obama_', 'obama_ge_')
            right_landmark = np.load(landmark_path)
            right_landmark = torch.FloatTensor(right_landmark.reshape(-1))
            # print ('=========================')
            # print ('real path: ' +  image_path)
            im = cv2.imread(image_path)
            if im is None:
                print (image_path)
                raise IOError
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.output_shape)
            im = self.transform(im)
            right_img = torch.FloatTensor(im)

          
            example_path =   '../image/musk1_region.jpg'
            example_landmark = np.load('../image/musk1.npy')
            # tp = ( np.dot(example_landmark.reshape(1,6), EIGVECS))[0,:].reshape(68,3)
            # tp = tp[:,:-1].reshape(-1)
            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)
            # print (right_landmark.size())

            return example_img, example_landmark, right_img,right_landmark

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)



class GRIDdataset1D_lstm_gt(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_16_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_16_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "new_16_full_gt_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':

            #load righ img
           
            image_path_root = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0])
                
            landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.train_data[index][0], self.train_data[index][0] + '_norm_lmarks.npy')

            current_frame_id =self.train_data[index][1]
            right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for jj in range(16):
                this_frame = current_frame_id + jj
                image_path =  os.path.join(image_path_root, '%05d.jpg'%this_frame)
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img[jj,:,:,:] = torch.FloatTensor(im)



            landmark = np.load(landmark_path)

            right_landmark = landmark[self.train_data[index][1] - 1 : self.train_data[index][1] + 15  ]
          
            right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))
           
            r = random.choice(
                [x for x in range(1,76)])
            example_path =   os.path.join(image_path_root, '%05d.jpg'%r)
            example_landmark = landmark[r - 1]

            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)

            return example_img, example_landmark, right_img,right_landmark
            
            
        elif self.train =='test':
            image_path_root = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0])
                
            landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/grid/data' , self.test_data[index][0], self.test_data[index][0] + '_norm_lmarks.npy')

            current_frame_id =self.test_data[index][1]
            right_img = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for jj in range(16):
                this_frame = current_frame_id + jj
                image_path =  os.path.join(image_path_root, '%05d.jpg'%this_frame)
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, self.output_shape)
                im = self.transform(im)
                right_img[jj,:,:,:] = torch.FloatTensor(im)



            landmark = np.load(landmark_path)

            right_landmark = landmark[self.test_data[index][1] - 1 : self.test_data[index][1] + 15  ]
          
            right_landmark = torch.FloatTensor(right_landmark.reshape(16,136))
           
            r = random.choice(
                [x for x in range(1,76)])
            example_path =   os.path.join(image_path_root, '%05d.jpg'%r)
            example_landmark = landmark[r - 1]

            example_landmark = torch.FloatTensor(example_landmark.reshape(-1))

            example_img = cv2.imread(example_path)
            example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
            example_img = cv2.resize(example_img, self.output_shape)
            example_img = self.transform(example_img)

            return example_img, example_landmark, right_img,right_landmark

       
    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)
class GRID_1D_lstm_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16  
        self.root_path = '/mnt/ssd0/dat/lchen63/grid/data'
        self.pca = torch.FloatTensor(np.load('../basics/U_grid.npy')[:,:6] ) 
        self.mean = torch.FloatTensor(np.load('../basics/mean_grid.npy'))
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

    
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            try:
                lmark_path = os.path.join(self.root_path , self.train_data[index][0] , self.train_data[index][0] + '_norm_lmarks.npy') 
                mfcc_path = os.path.join(self.root_path,  self.train_data[index][0],  self.train_data[index][0] +'_mfcc.npy') 
                
                lmark = np.load(lmark_path) * 5.0
                lmark = torch.FloatTensor(lmark)
                lmark = lmark - self.mean.expand_as(lmark)
                lmark = torch.mm(lmark,self.pca) 
                
                mfcc = np.load(mfcc_path)

                r = random.choice(
                    [x for x in range(6,50)])
                example_landmark =lmark[r,:]
                example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]
                
                mfccs = []
                for ind in range(1,17):
                    t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                    t_mfcc = torch.FloatTensor(t_mfcc)
                    mfccs.append(t_mfcc)
                mfccs = torch.stack(mfccs, dim = 0)
                landmark  =lmark[r+1 : r + 17,:]
               
                example_mfcc = torch.FloatTensor(example_mfcc)
                return example_landmark, example_mfcc, landmark, mfccs
            except:
                self.__getitem__(index + 1)
        if self.train=='test':
            lmark_path = os.path.join(self.root_path , self.test_data[index][0] , self.test_data[index][0] + '_norm_lmarks.npy') 
            mfcc_path = os.path.join(self.root_path,  self.test_data[index][0],  self.test_data[index][0] +'_mfcc.npy') 
            
            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca) 
            
            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,70)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]
           
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]
           
            example_mfcc = torch.FloatTensor(example_mfcc)
            
            return example_landmark, example_mfcc, landmark, mfccs
       
    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass


class GRID_1D_single_landmark_pca(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 16  
        self.root_path = '/mnt/ssd0/dat/lchen63/grid/data'
        self.pca = torch.FloatTensor(np.load('../basics/U_grid.npy')[:,:6] ) 
        self.mean = torch.FloatTensor(np.load('../basics/mean_grid.npy'))
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
            # try:

                lmark_path = os.path.join(self.root_path , self.train_data[index][0] , self.train_data[index][0] + '_norm_lmarks.npy') 
                mfcc_path = os.path.join(self.root_path,  self.train_data[index][0],  self.train_data[index][0] +'_mfcc.npy') 
                ind = self.train_data[index][1]
               
                lmark = np.load(lmark_path) * 5.0
                lmark = torch.FloatTensor(lmark)
                lmark = lmark - self.mean.expand_as(lmark)
                lmark = torch.mm(lmark,self.pca)

                
                mfcc = np.load(mfcc_path)

                r = random.choice(
                    [x for x in range(6,50)])
                example_landmark =lmark[r,:]
                t_mfcc =mfcc[(ind - 3)*4: (ind + 4)*4, 1:]

                t_mfcc = torch.FloatTensor(t_mfcc)
                landmark  =lmark[ind,:]
               
                return example_landmark, t_mfcc, landmark, t_mfcc
                
        if self.train=='test':
            lmark_path = os.path.join(self.root_path , self.test_data[index][0] , self.test_data[index][0] + '_norm_lmarks.npy') 
            mfcc_path = os.path.join(self.root_path,  self.test_data[index][0],  self.test_data[index][0] +'_mfcc.npy') 
            
            lmark = np.load(lmark_path) * 5.0
            lmark = torch.FloatTensor(lmark)
            lmark = lmark - self.mean.expand_as(lmark)
            lmark = torch.mm(lmark,self.pca) 
            
            mfcc = np.load(mfcc_path)

            r = random.choice(
                [x for x in range(3,70)])
            example_landmark =lmark[r,:]
            example_mfcc = mfcc[(r -3) * 4 : (r + 4) * 4, 1 :]
           
            mfccs = []
            for ind in range(1,17):
                t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r+1 : r + 17,:]
           
            example_mfcc = torch.FloatTensor(example_mfcc)
            
            return example_landmark, example_mfcc, landmark, mfccs
       
    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            pass
