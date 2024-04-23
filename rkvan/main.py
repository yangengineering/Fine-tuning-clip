import os
import argparse
from Trainer import MyTrainer
from utils.utils import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# CEC 150 setp 40
def set_parameters(parser):
    
    parser.add_argument('--annotation_train_exception_path', type=str, default='../data/exception_train.txt')
    parser.add_argument('--annotation_test_exception_path', type=str, default='../data/exception_test.txt')
    parser.add_argument('--annotation_train_normal_path', type=str, default='../data/normal_train.txt')
    parser.add_argument('--annotation_test_normal_path', type=str, default='../data/normal_test.txt')

    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.1, # 0.1 
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--scheduler', type=str, default='MSLR')
    parser.add_argument('--temperature', type=float, default=16) # 16

    
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--version', type=str, default='V1') 
    parser.add_argument('--arch_name', type=str, default='ViT-B/32') 
    parser.add_argument('--activate_branch', type=str, default='image_text')

    
    parser.add_argument('--exp_root', type=str, default='./outputs')
    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--seed', type=float, default=5)

    return parser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLIP')
    parser = set_parameters(parser)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    trainer = MyTrainer(args)
    print("TRAINING----------------")
    trainer.train()
    print("INFERENCE----------------")
    overall_acc,_,_,_ = trainer.test(load_best=True, write_csv=True)
    args.logger.info('Best overall accuracy: {}'.format(overall_acc))