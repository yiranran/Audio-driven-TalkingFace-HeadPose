import torch
from .base_model import BaseModel
from . import networks
from .memory_network import Memory_Network
import pdb

class MemorySeqModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unetac_adain_256', dataset_mode='aligned_feature_multi', direction='AtoB',Nw=3)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_mask', type=float, default=0.1, help='lambda mask loss')
            parser.add_argument('--lambda_mask_smooth', type=float, default=1e-5, help='lambda mask smooth loss')
        else:
            parser.add_argument('--test_use_gt', type=int, default=0, help='use gt feature in test')
        parser.add_argument('--attention', type=int, default=1, help='whether to use attention mechanism')
        parser.add_argument('--do_saturate_mask', action="store_true", default=False, help='do use mask_fake for mask_cyc')
        # for memory net
        parser.add_argument("--iden_feat_dim", type = int, default = 512)
        parser.add_argument("--spatial_feat_dim", type = int, default = 512)
        parser.add_argument("--mem_size", type = int, default = 30000)#982=819*1.2
        parser.add_argument("--alpha", type = float, default = 0.3)
        parser.add_argument("--top_k", type = int, default = 256)
        parser.add_argument("--iden_thres", type = float, default = 0.98)#0.5)
        parser.add_argument("--iden_feat_dir", type = str, default = 'arcface/iden_feat/')


        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.isTrain:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'mem']
            if self.opt.attention:
                self.loss_names += ['G_Att', 'G_Att_smooth', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.opt.Nw > 1:
            self.visual_names = ['real_A_0', 'real_A_1', 'real_A_2', 'fake_B', 'real_B']
        if self.opt.attention:
            self.visual_names += ['fake_B_img', 'fake_B_mask_vis']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'mem']
        else:  # during test time, only load G and mem
            self.model_names = ['G', 'mem']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc*opt.Nw, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, feat_dim=opt.iden_feat_dim)
        
        self.netmem = Memory_Network(mem_size = opt.mem_size, color_feat_dim = opt.iden_feat_dim, spatial_feat_dim = opt.spatial_feat_dim, top_k = opt.top_k, alpha = opt.alpha, gpu_ids = self.gpu_ids).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc*opt.Nw + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_mem = torch.optim.Adam(self.netmem.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_mem)
        self.replace = 0
        self.update = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # channel is input_nc * Nw
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # channel is output_nc
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # for memory net
        if self.isTrain or self.opt.test_use_gt:
            self.real_B_feat = input['B_feat' if AtoB else 'A_feat'].to(self.device)
        self.resnet_input = input['resnet_input'].to(self.device)
        self.idx = input['index'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.attention:
            if self.isTrain or self.opt.test_use_gt:
                self.fake_B_img, self.fake_B_mask = self.netG(self.real_A, self.real_B_feat)
            else:
                query = self.netmem(self.resnet_input)
                #pdb.set_trace()
                top1_feature, top1_index, topk_index = self.netmem.topk_feature(query, 1)
                top1_feature = top1_feature[:, 0, :]
                self.fake_B_img, self.fake_B_mask = self.netG(self.real_A, top1_feature)
            self.fake_B_mask = self._do_if_necessary_saturate_mask(self.fake_B_mask, saturate=self.opt.do_saturate_mask)
            self.fake_B = self.fake_B_mask * self.real_A[:,-self.opt.input_nc:] + (1 - self.fake_B_mask) * self.fake_B_img
            #print(torch.min(self.fake_B_mask), torch.max(self.fake_B_mask))
            self.fake_B_mask_vis = self.fake_B_mask * 2 - 1
        else:
            if self.isTrain or self.opt.test_use_gt:
                self.fake_B = self.netG(self.real_A, self.real_B_feat)
            else:
                query = self.netmem(self.resnet_input)
                top1_feature, _, _ = self.netmem.topk_feature(query, 1)
                top1_feature = top1_feature[:, 0, :]
                self.fake_B = self.netG(self.real_A, top1_feature)
        if self.opt.Nw > 1:
            self.real_A_0 = self.real_A[:,:self.opt.input_nc]
            self.real_A_1 = self.real_A[:,self.opt.input_nc:2*self.opt.input_nc]
            self.real_A_2 = self.real_A[:,-self.opt.input_nc:]
    
    def backward_mem(self):
        #print(self.image_paths)
        resnet_feature = self.netmem(self.resnet_input)
        self.loss_mem = self.netmem.unsupervised_loss(resnet_feature, self.real_B_feat, self.opt.iden_thres)
        self.loss_mem.backward()
    
    def update_mem(self):
        with torch.no_grad():
            resnet_feature = self.netmem(self.resnet_input)
            replace = self.netmem.memory_update(resnet_feature, self.real_B_feat, self.opt.iden_thres, self.idx)
            if replace:
                self.replace += 1
            else:
                self.update += 1

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Loss for attention mask
        if self.opt.attention:
            # the attention mask can easily saturate to 1, which makes that generator has no effect
            self.loss_G_Att = torch.mean(self.fake_B_mask) * self.opt.lambda_mask
            # to enforce smooth spatial color transformation
            self.loss_G_Att_smooth = self._compute_loss_smooth(self.fake_B_mask) * self.opt.lambda_mask_smooth
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.opt.attention:
            self.loss_G += self.loss_G_Att + self.loss_G_Att_smooth
        self.loss_G.backward()

    def optimize_parameters(self):
        # update mem
        self.optimizer_mem.zero_grad()
        self.backward_mem()
        self.optimizer_mem.step()
        self.update_mem()

        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m
