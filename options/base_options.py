import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--dataroot', default='./dataset',
                            help='path to images (should have subfolders)')
        parser.add_argument('--name', type=str, default='MTL',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='-1',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for another GPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='MTL',
                            help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=1,
                            help='input image channels')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='utput image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='n_layers',
                            help='specify discriminator architecture')
        parser.add_argument('--netG', type=str, default='UNet',
                            help='specify generator architecture')
        parser.add_argument('--deconv', type=int, default=0, help='specify deconv times ')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='batch',
                            help='instance normalization or batch normalization')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', default="True", help='no dropout for the generator')
        # parser.add_argument('--lambda_identity', type=int, default=0, help='just needed for preserving the color.')
        parser.add_argument('--dataset_mode', type=str, default='aligned',
                            help='chooses how datasets are loaded.')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=128, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=128, help='then crop to this size')
        parser.add_argument('--rotation_degrees', type=int, default=10, help='the rotation range')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='none',
                            help='scaling and cropping of images at load time')
        parser.add_argument('--flip', type=int, default=0,
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=128,
                            help='display window size for both visdom and HTML')

        parser.add_argument('--gray_low_dataB', type=float, default=0, help='set lowest grayscale of dataB')
        parser.add_argument('--gray_high_dataB', type=float, default=1.0, help='set highest grayscale of dataB')
        parser.add_argument('--gray_low_dataA', type=float, default=0, help='set lowest grayscale of dataA')
        parser.add_argument('--gray_high_dataA', type=float, default=1.0, help='set highest grayscale of dataA')

        # parser.add_argument('--img_spacing_0', default=3.5, type=float, help='save image spacing_0')
        # parser.add_argument('--img_spacing_1', default=3.5, type=float, help='save image spacing_1')
        # parser.add_argument('--img_spacing_2', default=3.5, type=float, help='save image spacing_2')

        parser.add_argument('--img_spacing', nargs='+', type=float, help='save image with pixel spacing')

        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default=0,
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        parser.add_argument('--ifRotate_n90', type=int, default=0, help='Rotate n * 90° for data_augmented')
        parser.add_argument('--ifCrop', type=int, default=0, help='Random Crop for data_augmented')
        parser.add_argument('--ifGAN', type=int, default=0, help='use GAN or only generator')
        parser.add_argument('--attention_form', type=str, default='CBAM', help='use AG or CBAM attention')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        # print(dataset_option_setter)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        # print(vars(opt))
        # print(opt)
        # print(vars(opt).items())
        # print(sorted(vars(opt).items()))
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            # print(k," : ",default)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        # print(opt.name,
        #       opt.suffix,
        #       opt.checkpoints_dir,
        #       opt.dataset_mode,
        #       opt.gpu_ids,
        #       opt.isTrain,
        #       opt.model,sep="\n"
        #         )
        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
