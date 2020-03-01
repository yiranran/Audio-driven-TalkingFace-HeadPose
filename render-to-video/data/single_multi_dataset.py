from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os.path
import torch


class SingleMultiDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        imglistA = 'datasets/list/%s/%s.txt' % (opt.phase+'Single', opt.dataroot)
        if not os.path.exists(imglistA):
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        else:
            self.A_paths = open(imglistA, 'r').read().splitlines()
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.Nw = self.opt.Nw

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')

        # apply the same transform to both A and resnet_input
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=self.opt.resizemethod)
        A = A_transform(A_img)
        As = torch.zeros((self.input_nc * self.Nw, self.opt.crop_size, self.opt.crop_size))
        As[-self.input_nc:] = A
        frame = os.path.basename(A_path).split('_')[0]
        ext = os.path.basename(A_path).split('_')[1]
        frameno = int(frame)
        for i in range(1,self.Nw):
            # read frameno-i frame
            path1 = A_path.replace(frame+'_blend','%05d_blend'%(frameno-i))
            A = Image.open(path1).convert('RGB')
            # store in Nw-i's
            As[-(i+1)*self.input_nc:-i*self.input_nc] = A_transform(A)
        item = {'A': As, 'A_paths': A_path}

        if self.opt.use_memory:
            resnet_transform = get_transform(self.opt, transform_params, grayscale=False, resnet=True, method=self.opt.resizemethod)
            resnet_input = resnet_transform(A_img)
            item['resnet_input'] = resnet_input
        
        return item

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
