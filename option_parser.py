import argparse
import torch
from utils import utils
import os


class OptionParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def parse_args(self):
        opt = self.parser.parse_args()
        str_ids = str(opt.gpu_ids)
        opt.gpu_ids = []
        for str_id in str_ids.split(','):
            opt.gpu_ids.append(int(str_id))
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('-------- [INFO] Options --------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

        expr_dir = os.path.join(opt.ckpt_dir, opt.model)
        utils.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(' [INFO] Options\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
        print('------------- END -------------')
        return opt

    def set_arguments(self):
        # training options
        self.parser.add_argument('--dataset', type=str, default='MNIST', help='name of dataset. MNIST default')
        self.parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        self.parser.add_argument('--num_preprocess_workers', type=int, default=2, help='num preprocess workers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids ex) 0,1,2')
        self.parser.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='checkpoint dir')
        self.parser.add_argument('--model', type=str, default='NIMA', help='name of model')
        self.parser.add_argument('--epoch', type=int, default=20, help='epoch')
        self.parser.add_argument('--initial_size', type=int, default=784, help='initial tensor size')
        self.parser.add_argument('--label_size', type=int, default=10, help='label size')
        self.parser.add_argument('--eps', type=float, default=1e-8, help='eps')

        # visualize options
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.ckpt_dir]/[opt.model]/web/')

        # input parameters
        self.parser.add_argument('--img_path', type=str, default='./data')
        self.parser.add_argument('--test_path', type=str, default='./test')
        self.parser.add_argument('--test_image_dir', type=str, default='reviews/bad')
        self.parser.add_argument('--train_csv_file', type=str, default='./ann_train.csv')
        self.parser.add_argument('--val_csv_file', type=str, default='./ann_val.csv')
        self.parser.add_argument('--test_csv_file', type=str, default='./ann_test.csv')

        # training parameters
        self.parser.add_argument('--is_train', type=int, default=1)
        self.parser.add_argument('--is_validation', type=int, default=1)
        self.parser.add_argument('--conv_base_lr', type=float, default=3e-7)
        self.parser.add_argument('--dense_lr', type=float, default=3e-6)
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.95)
        self.parser.add_argument('--lr_decay_freq', type=int, default=10)
        self.parser.add_argument('--train_batch_size', type=int, default=128)
        self.parser.add_argument('--val_batch_size', type=int, default=128)
        self.parser.add_argument('--test_batch_size', type=int, default=1)
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument('--epochs', type=int, default=100)

        # misc
        self.parser.add_argument('--multi_gpu', type=bool, default=False)
        self.parser.add_argument('--warm_start', type=bool, default=False)
        self.parser.add_argument('--warm_start_epoch', type=int, default=0)
        self.parser.add_argument('--early_stopping_patience', type=int, default=5)
        self.parser.add_argument('--save_fig', type=bool, default=False)
