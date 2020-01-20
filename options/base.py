import argparse
import os
import time

from utils import util


class BaseOptions:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='troy_dev', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--num_workers', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per datasets. If the datasets directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='None',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--padding_type', type=str, default='reflect', help='# of input image channels')
        self.parser.add_argument('--configure_path', type=str, default='configures', help='configure path.')
        self.parser.add_argument('--configure_file', type=str, default='train', help='configure files.')
        self.parser.add_argument('--suffix', type=str, default='default', help='configure files.')
        self.parser.add_argument('--no_html', type=bool, default=False, help='configure files.')
        self.parser.add_argument('--save_epoch', type=int, default=10, help='epochs to save.')
        self.opt = self.parser.parse_args()
        # down-sampling times
        self.initialized = True
        # build configure file.
        self.opt.configure_file = os.path.join(self.opt.configure_path, self.opt.configure_file + ".yaml")



    def parse(self, configure=None):
        if not self.initialized:
            self.initialize()

        if configure is not None:
            for k, v in configure.items():
                self.opt.__setattr__(k, v)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        # save to the disk
        time_str = time.strftime("Trail#%j%H%M%S", time.localtime(time.time()))
        exp_name = '-'.join([self.opt.model, self.opt.dataset, str(self.opt.lr), self.opt.optimizer, self.opt.suffix, time_str])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, exp_name)
        self.opt.__setattr__('expr_dir', expr_dir)
        self.opt.__setattr__('data_gen', self.opt.checkpoints_dir)
        self.opt.__setattr__('resume', expr_dir)
        util.make_dirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        print('=> make opt file in :\n\t{}'.format(os.path.abspath(file_name)))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
