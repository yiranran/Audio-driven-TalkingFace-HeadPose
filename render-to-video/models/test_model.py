from .base_model import BaseModel
from . import networks
from .memory_network import Memory_Network


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(norm='batch')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--attention', type=int, default=0, help='whether to use attention mechanism')
        parser.add_argument('--do_saturate_mask', action="store_true", default=False, help='do use mask_fake for mask_cyc')
        # for memory net
        parser.add_argument("--use_memory", type = int, default = 0)
        parser.add_argument("--iden_feat_dim", type = int, default = 512)
        parser.add_argument("--spatial_feat_dim", type = int, default = 512)
        parser.add_argument("--mem_size", type = int, default = 30000)#982=819*1.2
        parser.add_argument("--alpha", type = float, default = 0.3)
        parser.add_argument("--top_k", type = int, default = 256)
        parser.add_argument("--iden_thres", type = float, default = 0.5)
        parser.add_argument("--save2", type = int, default = 0)
        parser.add_argument("--fixindex", type = int, default = -1)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        if self.opt.Nw > 1:
            self.visual_names = ['real_A_0', 'real_A_1', 'real_A_2', 'fake']
        if self.opt.attention:
            self.visual_names += ['fake_B_img', 'fake_B_mask_vis']
        if self.opt.save2:
            if self.opt.Nw == 1:
                self.visual_names = ['real', 'fake']
            else:
                self.visual_names = ['real_A_0', 'real_A_1', 'real_A_2', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc*opt.Nw, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.opt.use_memory:
            self.netmem = Memory_Network(mem_size = opt.mem_size, color_feat_dim = opt.iden_feat_dim, spatial_feat_dim = opt.spatial_feat_dim, top_k = opt.top_k, alpha = opt.alpha, gpu_ids = self.gpu_ids)
            self.model_names.append('mem')

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        # for memory net
        if self.opt.use_memory:
            self.resnet_input = input['resnet_input'].to(self.device)

    def forward(self):
        """Run forward pass."""
        if not self.opt.attention:
            self.fake = self.netG(self.real)  # G(A)
        else:
            if not self.opt.use_memory:
                self.fake_B_img, self.fake_B_mask = self.netG(self.real)
            else:
                if self.opt.fixindex != -1:
                    top1_idx = self.opt.fixindex
                    top1_feature, top1_index = self.netmem.get_feature(top1_idx, self.real.size()[0])
                    #print(top1_index, top1_idx)
                else:
                    query = self.netmem(self.resnet_input)
                    top1_feature, top1_index, top1_idx = self.netmem.topk_feature(query, 1)
                    top1_feature = top1_feature[:, 0, :]
                    #print(top1_index, top1_idx)
                self.fake_B_img, self.fake_B_mask = self.netG(self.real, top1_feature)
            self.fake_B_mask = self._do_if_necessary_saturate_mask(self.fake_B_mask, saturate=self.opt.do_saturate_mask)
            self.fake = self.fake_B_mask * self.real[:,-self.opt.input_nc:] + (1 - self.fake_B_mask) * self.fake_B_img
            self.fake_B_mask_vis = self.fake_B_mask * 2 - 1
        if self.opt.Nw > 1:
            self.real_A_0 = self.real[:,:self.opt.input_nc]
            self.real_A_1 = self.real[:,self.opt.input_nc:2*self.opt.input_nc]
            self.real_A_2 = self.real[:,2*self.opt.input_nc:3*self.opt.input_nc]
    
    def forward_getfeat(self):
        query = self.netmem(self.resnet_input)
        top1_feature, top1_index, top1_idx = self.netmem.topk_feature(query, 1)
        return top1_feature
    
    def forward_getfeatk(self, k):
        query = self.netmem(self.resnet_input)
        topk_feature, topk_index, topk_idx = self.netmem.topk_feature(query, k)
        return topk_feature
    
    def forward_withfeat(self,feat):
        self.fake_B_img, self.fake_B_mask = self.netG(self.real, feat)
        self.fake_B_mask = self._do_if_necessary_saturate_mask(self.fake_B_mask, saturate=self.opt.do_saturate_mask)
        self.fake = self.fake_B_mask * self.real[:,-self.opt.input_nc:] + (1 - self.fake_B_mask) * self.fake_B_img
        self.fake_B_mask_vis = self.fake_B_mask * 2 - 1
        if self.opt.Nw > 1:
            self.real_A_0 = self.real[:,:self.opt.input_nc]
            self.real_A_1 = self.real[:,self.opt.input_nc:2*self.opt.input_nc]
            self.real_A_2 = self.real[:,2*self.opt.input_nc:3*self.opt.input_nc]


    def optimize_parameters(self):
        """No optimization for test model."""
        pass
    
    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m