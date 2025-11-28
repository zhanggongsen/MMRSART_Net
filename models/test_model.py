from .base_model import BaseModel
from . import networks
import torch.nn as nn


class TestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='aligned')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        return parser

    def __init__(self, opt):
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.attention_form = opt.attention_form
        self.GNet = opt.netG
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A','real_B', 'fake_B','real_C', 'fake_C']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      self.attention_form)


        # assigns the model to self.netG_[suffix] so that it can be loaded
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    # def set_input(self, input):
    #     self.real_A = input['A'].to(self.device)
    #     self.real_B = input['B'].to(self.device)
    #     self.real_C = input['C'].to(self.device)
    #     self.image_paths = input['A_paths']
    #
    # def forward(self):
    #     """Run forward pass."""
    #     self.fake_B,self.fake_C = self.netG(self.real_A)  # G(A)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)
        self.real_E = input['E'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # self.fake_B = self.netG(self.real_A)  # G(A)
        if self.GNet == "Swin_Unetr":
            self.fake_B = nn.Tanh(self.netG(self.real_A))  # G(A)
        else:
            self.fake_B, self.fake_C = self.netG(self.real_A,self.real_D,self.real_E)  # B,C=G(A)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
