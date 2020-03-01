import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch


class AlignedFeatureMultiDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        imglistA = 'datasets/list/%s/%s.txt' % (opt.phase+'A', opt.dataroot)
        imglistB = 'datasets/list/%s/%s.txt' % (opt.phase+'B', opt.dataroot)

        if not os.path.exists(imglistA) or not os.path.exists(imglistB):
            self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
            self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        else:
            self.AB_paths = open(imglistA, 'r').read().splitlines()
            self.AB_paths2 = open(imglistB, 'r').read().splitlines()

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.Nw = self.opt.Nw

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # by default A and B are from 2 png
        AB_path = self.AB_paths[index]
        A = Image.open(AB_path).convert('RGB')
        AB_path2 = self.AB_paths2[index]
        B = Image.open(AB_path2).convert('RGB')
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=self.opt.resizemethod)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1),method=self.opt.resizemethod)
        resnet_transform = get_transform(self.opt, transform_params, grayscale=False, resnet=True, method=self.opt.resizemethod)

        imA = A
        A = A_transform(A)
        B = B_transform(B)
        resnet_input = resnet_transform(imA)

        As = torch.zeros((self.input_nc * self.Nw, self.opt.crop_size, self.opt.crop_size))
        As[-self.input_nc:] = A
        frame = os.path.basename(AB_path).split('_')[0]
        frameno = int(frame[5:])
        for i in range(1,self.Nw):
            # read frameno-i frame
            path1 = AB_path.replace(frame,'frame%d'%(frameno-i))
            A = Image.open(path1).convert('RGB')
            # store in Nw-i's
            As[-(i+1)*self.input_nc:-i*self.input_nc] = A_transform(A)

        item = {'A': As, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path2}
        item['resnet_input'] = resnet_input
        item['index'] = np.array(([index+0.0])).astype(np.float32)[0]

        if self.opt.isTrain or self.opt.test_use_gt:
            AB_path2 = AB_path2.replace('_input2','')
            ss = AB_path2.split('/')
            if self.opt.dataroot != '300vw_win3' and self.opt.dataroot != 'lrwnewrender_win3' and ss[-3] != '19_news':
                B_feat = np.load(os.path.join(self.opt.iden_feat_dir,ss[-2],ss[-1][:-4]+'.npy'))
            elif self.opt.dataroot == '300vw_win3':
                B_feat = np.load(os.path.join(self.opt.iden_feat_dir,ss[-3],ss[-2],ss[-1][:-4]+'.npy'))
            elif self.opt.dataroot == 'lrwnewrender_win3':
                B_feat = np.load(os.path.join(self.opt.iden_feat_dir,ss[-4],ss[-3],ss[-2],'frame14.npy'))
            elif ss[-3] == '19_news':
                B_feat = np.load(os.path.join(self.opt.iden_feat_dir,ss[-3],ss[-2],ss[-1][:-4]+'.npy'))
            item['B_feat'] = B_feat

        return item

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
