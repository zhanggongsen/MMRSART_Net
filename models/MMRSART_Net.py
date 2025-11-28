import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.autograd import Variable
from monai.losses import SSIMLoss


class MMRSART_Net(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='UNet_3Plus', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--ifL1Loss', type=int, default=0, help='use L1Loss for Net_G')
            parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')
            parser.add_argument('--ifMSELoss', type=int, default=0, help='use MSELoss for Net_G')
            parser.add_argument('--lambda_MSE', type=float, default=1, help='weight for MSE loss')
            parser.add_argument('--ifSSIMLoss', type=int, default=0, help='use SSIMLoss for Net_G')
            parser.add_argument('--lambda_SSIM', type=float, default=1, help='weight for SSIM loss')
            parser.add_argument('--ifGDLoss', type=int, default=0, help='use GDLoss for Net_G')
            parser.add_argument('--lambda_GD', type=float, default=1, help='weight for GD loss')
            parser.add_argument('--ifBCELoss', type=int, default=0, help='use BCELoss for Net_G')
            parser.add_argument('--lambda_BCE', type=float, default=1, help='weight for BCELoss loss')
            parser.add_argument('--ifDiceLoss', type=int, default=0, help='use DiceLoss for Net_G')
            parser.add_argument('--lambda_Dice', type=float, default=1, help='weight for Dice loss')
            # parser.add_argument('--ifMPDLoss', type=int, default=0, help='use MPDLoss for Net_G')
            # parser.add_argument('--MPD_P', type=float, default=1.5, help='MPDLoss_P')
            # parser.add_argument('--lambda_MPD', type=float, default=80.0, help='weight for MPD loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.GNet = opt.netG
        self.ifGAN = opt.ifGAN
        self.gan_mode = opt.gan_mode
        self.attention_form = opt.attention_form
        self.ifL1Loss = opt.ifL1Loss
        self.lambda_L1 = opt.lambda_L1
        self.ifMSELoss = opt.ifMSELoss
        self.lambda_MSE = opt.lambda_MSE
        self.ifSSIMLoss = opt.ifSSIMLoss
        self.lambda_SSIM = opt.lambda_SSIM
        self.ifGDLoss = opt.ifGDLoss
        self.lambda_GD = opt.lambda_GD
        self.ifBCELoss = opt.ifBCELoss
        self.lambda_BCE = opt.lambda_BCE
        self.ifDiceLoss = opt.ifDiceLoss
        self.lambda_Dice = opt.lambda_Dice
        # self.ifMPDLoss = opt.ifMPDLoss
        # self.MPD_P = opt.MPD_P
        # self.lambda_MPD = opt.lambda_MPD
        if self.ifGAN == 1:
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'D']
        else:
            self.loss_names = []
        if self.ifL1Loss == 1:
            self.loss_names.append('G_L1Loss')
        if self.ifMSELoss == 1:
            self.loss_names.append('G_MSELoss')
        if self.ifSSIMLoss == 1:
            self.loss_names.append('G_SSIMLoss')
        if self.ifGDLoss == 1:
            self.loss_names.append("G_GDLoss")
        if self.ifBCELoss == 1:
            self.loss_names.append('G_BCELoss')
        if self.ifDiceLoss == 1:
            self.loss_names.append('G_DiceLoss')
        else:
            pass
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B', 'real_B', "fake_C", "real_C"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and (self.ifGAN == 1):
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.deconv)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      self.attention_form)
        if self.isTrain and (
                self.ifGAN == 1):  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, 'batch', opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            pass

        if self.isTrain:
            # define loss functions
            self.L1LossFuc = torch.nn.L1Loss()
            # self.L1LossFuc = networks.L1_Charbonnier_loss().to(self.device) # Charbonnier_loss是对L1 Loss的改进
            # self.L1LossFuc = torch.nn.SmoothL1Loss(reduction='mean')
            self.MSELossFuc = torch.nn.MSELoss(reduction='mean')
            self.SSIMLossFuc = SSIMLoss(spatial_dims=3)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if (self.ifGAN == 1):
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                if self.gan_mode == 'wgangp':
                    self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=opt.lr)
                    self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr)
                else:
                    self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
                self.optimizers.append(self.optimizer_G)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)
        self.real_E = input['E'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)
        if self.GNet == "Swin_Unetr":
            self.fake_B = nn.Tanh(self.netG(self.real_A))  # G(A)
        else:
            self.fake_B, self.fake_C = self.netG(self.real_A,self.real_D,self.real_E)  # B,C=G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
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
        if self.ifL1Loss == 1:
            self.loss_G_L1Loss = self.L1LossFuc(self.fake_B, self.real_B)
        else:
            self.loss_G_L1Loss = 0

        if self.ifMSELoss == 1:
            self.loss_G_MSELoss = self.MSELossFuc(self.fake_B, self.real_B)
        else:
            self.loss_G_MSELoss = 0

        if self.ifSSIMLoss == 1:
            self.loss_G_SSIMLoss = self.SSIMLossFuc(self.fake_B, self.real_B)
        else:
            self.loss_G_SSIMLoss = 0
        if self.ifGDLoss == 1:
            self.loss_G_GDLoss = self.GDLossFuc(self.fake_B, self.real_B)
        else:
            self.loss_G_GDLoss = 0

        if self.ifBCELoss == 1:
            self.BCELL = nn.BCELoss()
            self.loss_G_BCELoss = self.BCELL(self.fake_C, self.real_C)
        else:
            self.loss_G_BCELoss = 0

        if self.ifDiceLoss == 1:
            self.loss_G_DiceLoss = self.DiceLoss(self.fake_C, self.real_C)
        else:
            self.loss_G_DiceLoss = 0

        if (self.ifGAN == 1):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G = self.loss_G_GAN + self.loss_G_L1Loss * self.lambda_L1 + self.loss_G_MSELoss * self.lambda_MSE + self.loss_G_SSIMLoss * self.lambda_SSIM + self.loss_G_GDLoss * self.lambda_GD + self.loss_G_BCELoss * self.lambda_BCE + self.loss_G_DiceLoss * self.lambda_Dice
        else:
            self.loss_G = self.loss_G_L1Loss * self.lambda_L1 + self.loss_G_MSELoss * self.lambda_MSE + self.loss_G_SSIMLoss * self.lambda_SSIM + self.loss_G_GDLoss * self.lambda_GD + self.loss_G_BCELoss * self.lambda_BCE + self.loss_G_DiceLoss * self.lambda_Dice
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        if (self.ifGAN == 1):
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            if self.gan_mode == 'wgangp':
                # clip D weights between -0.01, 0.01
                for p in self.netD.parameters():
                    p.data.clamp_(-0.01, 0.01)
            self.set_requires_grad(self.netD, False)
        else:
            pass
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def GDLossFuc(self, a, b):
        # print(a.size())   #torch.Size([16,1, 1, 256, 256])
        GD = 0
        for i in range(a.shape[0]):
            A = a[i][0]
            B = b[i][0]
            if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):

                deltaA_x = torch.abs(A[:(A.shape[0] - 1), :, :] - A[1:, :, :])
                deltaB_x = torch.abs(B[:(B.shape[0] - 1), :, :] - B[1:, :, :])
                deltax = (deltaA_x - deltaB_x) * (deltaA_x - deltaB_x)
                GD_x = deltax.mean()

                deltaA_y = torch.abs(A[:, :(A.shape[1] - 1), :] - A[:, 1:, :])
                deltaB_y = torch.abs(B[:, :(B.shape[1] - 1), :] - B[:, 1:, :])
                deltay = (deltaA_y - deltaB_y) * (deltaA_y - deltaB_y)
                GD_y = deltay.mean()

                deltaA_z = torch.abs(A[:, :, :(A.shape[2] - 1)] - A[:, :, 1:])
                deltaB_z = torch.abs(B[:, :, :(B.shape[2] - 1)] - A[:, :, 1:])
                deltaz = (deltaA_z - deltaB_z) * (deltaA_z - deltaB_z)
                GD_z = deltaz.mean()
                GD += GD_x + GD_y + GD_z
            else:
                raise TypeError("Your input needs to be tensor")
        return GD / a.shape[0]

    def DiceLoss(self, prediction, target):
        smooth = 1e-5
        # i_flat = torch.sigmoid(prediction).view(-1)
        i_flat = prediction.view(-1)
        t_flat = target.view(-1)

        intersection = (i_flat * t_flat).sum()
        domina = i_flat.sum() + t_flat.sum()
        # DSCLoss=1-(2. * intersection) / domina
        DSCLoss = 1 - (((2. * intersection) + smooth) / (domina + smooth))
        # print("i_flat{},t_flat{},intersection{},domina{},DSCLoss{}".format(i_flat.sum(), t_flat.sum(), intersection,
        #                                                                    domina, DSCLoss))
        # return 1 - ((2. * intersection + smooth) / (domina + smooth))
        return DSCLoss