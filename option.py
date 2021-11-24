# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp_name',     required=True,                      help='experiment name')
parser.add_argument('--description',        default=None,                       help='description of current experiment')
parser.add_argument('--comment',            default=None,                       help='comment for the current experiment')

parser.add_argument('--data',               default='CelebA-HQ',    type=str,   help='CelebA-HQ, utkface, or imagenet')
parser.add_argument('--n_class',            default=2,              type=int,   help='number of classes')
parser.add_argument('--input_size',         default=224,            type=int,   help='input size')
parser.add_argument('--batch_size',         default=32,             type=int,   help='mini-batch size')
parser.add_argument('--momentum',           default=0.9,            type=float, help='optimizer momentum')
parser.add_argument('--lr',                 default=0.001,          type=float, help='initial learning rate')
parser.add_argument('--lr_decay_rate',      default=0.1,            type=float, help='lr decay rate')
parser.add_argument('--lr_decay_period',    default=10,             type=int,   help='lr decay period')
parser.add_argument('--lr_scheduler',       default='step',                     help='learning rate scheduler - step, cosine')
parser.add_argument('--optim',              default='Adam',                     help='optimizer - Adam, AdamP')
parser.add_argument('--weight_decay',       default=0.0005,         type=float, help='optimizer weight decay')
parser.add_argument('--max_step',           default=10,             type=int,   help='maximum step for training')
parser.add_argument('--depth',              default=20,             type=int,   help='depth of network')
parser.add_argument('--data_var',           default=0.0,            type=float, help='variance for data distribution')
parser.add_argument('--seed',               default=2,              type=int,   help='seed index')

parser.add_argument('--checkpoint',         default=None,                       help='checkpoint to resume')
parser.add_argument('--checkpoint_orth',    default=None,                       help='checkpoint to resume for orth')
parser.add_argument('--log_step',           default=50,             type=int,   help='step for logging in iteration')
parser.add_argument('--save_step',          default=1,              type=int,   help='step for saving in epoch')
parser.add_argument('--data_dir',           default='./',                       help='data directory')
parser.add_argument('--save_dir',           default='./',                       help='save directory for checkpoint')
parser.add_argument('--data_split',         default='train',                    help='data split to use')
parser.add_argument('--use_pretrain',       default=False,                      help='whether it use pre-trained parameters if exists')
parser.add_argument('--imagenet_pretrain',  action='store_true',                help='whether it train base_model or unlearning')

parser.add_argument('--random_seed',                                type=int,   help='random seed')
parser.add_argument('--num_workers',        default=0,              type=int,   help='number of workers in data dataset')
parser.add_argument('--cudnn_benchmark',    default=True,           type=bool,  help='cuDNN benchmark')

parser.add_argument('--cuda',               action='store_true',                help='enables cuda')
parser.add_argument('--gpu',                default='3',                        help='which number of gpu used')
parser.add_argument('-d', '--debug',        action='store_true',                help='debug mode')
parser.add_argument('--is_train',           action='store_true',                help='whether it is training')
parser.add_argument('--is_valid',           action='store_true',                help='whether it is validation')
parser.add_argument('--ubnet',              action='store_true',                help='whether using ubnet')

parser.add_argument('--bias_type',          default=None,                       help='training bias: ub1 or ub2')
parser.add_argument('--cls_type',           default=None,                       help='type for utkface classification, gender_skintone or skintone_gender')
parser.add_argument('--model',              required=True,                      help='which model is used for backbone')
parser.add_argument('--mid_ch',             default=64,             type=int,   help='middle(trans) layer number of channels')


def get_option():
    option = parser.parse_args()
    return option
