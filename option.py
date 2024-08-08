import argparse
import utility
import numpy as np

parser = argparse.ArgumentParser(description='DRN')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--data_dir', type=str, default='data_path',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='simulation_pair_data',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='pmri',
                    help='test dataset name')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--model', help='SCUNET', required=True)
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--loss', type=str, default='L1',
                    help='loss function configuration, L1|SSIM')
parser.add_argument('--save', type=str, default='/home/experiment/lf_finetune/',
                    help='file name to save')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

args = parser.parse_args()

utility.init_model(args)


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
