import torch
from torch.backends import cudnn

import os, random, json

from option import get_option
from trainer import Trainer
from utils import save_option

import torchvision.transforms as transforms
import torch.utils.data as data

from loader.imagenet_dataset import get_imagenet_dataloader
from loader.utkface_dataset import UTKFaceDataset
from loader.celebA_dataset import CelebA_HQ
from utils import logger_setting

def backend_setting(option):
    log_dir = os.path.join(option.save_dir, option.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = option.gpu

    if option.random_seed is None:
        option.random_seed = random.randint(1,10000)
    torch.manual_seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        print('WARNING: GPU is available, but not use it')

    if option.cuda:
        torch.cuda.manual_seed_all(option.random_seed)
        cudnn.benchmark = option.cudnn_benchmark

def main():
    option = get_option()
    backend_setting(option)

    logger = logger_setting(option.exp_name, option.save_dir, option.debug)
    with open (os.path.join(option.save_dir, option.exp_name,'params.json'), 'w') as outfile:
        json.dump(option.__dict__, outfile, indent=4, sort_keys=True)

    trainer = Trainer(option, logger)

    if option.data == 'utkface':
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        valid_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if option.bias_type == 'ub1':
            train_bias = 'ub1'
            valid_bias = 'ub2'
        elif option.bias_type == 'ub2':
            train_bias = 'ub2'
            valid_bias = 'ub1'

        var = option.data_var

        train_dataset = UTKFaceDataset(data_dir=os.path.join(option.data_dir, 'utkface/utkcropped'),
                                       json_dir=os.path.join(option.data_dir, 'utkface/'),
                                       split=train_bias, cls_type=option.cls_type,
                                       var=var, transforms=train_transforms)
        train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=option.batch_size,
                                    shuffle=True,
                                    num_workers=option.num_workers)

        valid_test_dataset = UTKFaceDataset(data_dir=os.path.join(option.data_dir, 'utkface/utkcropped'),
                                            json_dir=os.path.join(option.data_dir, 'utkface/'),
                                            split='test', cls_type=option.cls_type,
                                            var=var, transforms=valid_transforms)
        valid_bias_dataset = UTKFaceDataset(data_dir=os.path.join(option.data_dir, 'utkface/utkcropped'),
                                            json_dir=os.path.join(option.data_dir, 'utkface/'),
                                            split=valid_bias, cls_type=option.cls_type,
                                            var=var, transforms=valid_transforms)

        valid_test_loader = data.DataLoader(dataset=valid_test_dataset,
                                    batch_size=option.batch_size,
                                    shuffle=True,
                                    num_workers=option.num_workers)
        valid_bias_loader = data.DataLoader(dataset=valid_bias_dataset,
                                            batch_size=option.batch_size,
                                            shuffle=True,
                                            num_workers=option.num_workers)

        print(f"train_dataset: {len(train_dataset)} | valid_test_dataset: {len(valid_test_dataset)}"
              f" | valid_bias_dataset: {len(valid_bias_dataset)}")

        valid_types = [f'{train_bias.upper()}-Test', f'{train_bias.upper()}-{valid_bias.upper()}']
        loaders = [valid_test_loader, valid_bias_loader]

        if option.is_train:
            trainer.train(train_loader, val_loaders=loaders, val_types=valid_types)
        else:
            for val_type, val_loader in zip(valid_types, loaders):
                trainer._validate(val_loader, step=0, valid_type=val_type)

    elif option.data == 'imagenet':
        if option.is_train:
            train_loader = get_imagenet_dataloader(root=os.path.join(option.data_dir, 'imagenet/train'),
                                                   batch_size=option.batch_size,
                                                   train=True,
                                                   val_data=None)
        valid_loader = get_imagenet_dataloader(root=os.path.join(option.data_dir, 'imagenet/val'),
                                               batch_size=option.batch_size,
                                               train=False,
                                               val_data='ImageNet')

        valid_types = ['biased', 'unbiased']
        loaders = [valid_loader, valid_loader]

        if option.is_train:
            trainer.train(train_loader, val_loaders=loaders, val_types=valid_types)
        else:
            for idx, (val_type, val_loader) in enumerate(zip(valid_types, loaders)):
                trainer._validate_imagenet(val_loader, 0, valid_type=val_type)

    elif option.data == 'CelebA-HQ':
        train_dataset = CelebA_HQ(root=os.path.join(option.data_dir, 'celebA'), txt_file='dataset/CelebA-HQ/train.txt')
        ub1_valid_dataset = CelebA_HQ(root=os.path.join(option.data_dir, 'celebA'), txt_file='dataset/CelebA-HQ/ub1_val.txt')
        ub2_valid_dataset = CelebA_HQ(root=os.path.join(option.data_dir, 'celebA'), txt_file='dataset/CelebA-HQ/ub2_val.txt')

        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=option.batch_size,
                                       num_workers=option.num_workers)

        ub1_valid_loader = data.DataLoader(dataset=ub1_valid_dataset,
                                           batch_size=option.batch_size,
                                           num_workers=option.num_workers)

        ub2_valid_loader = data.DataLoader(dataset=ub2_valid_dataset,
                                           batch_size=option.batch_size,
                                           num_workers=option.num_workers)

        print(
            f"train_dataset: {len(train_dataset)} | eb1_valid_dataset: {len(ub1_valid_dataset)} | eb2_valid_dataset: {len(ub2_valid_dataset)}")
        print(
            f"train_loader: {len(train_loader)} | eb1_valid_loader: {len(ub1_valid_loader)} | eb2_valid_loader: {len(ub2_valid_loader)}")
        valid_types = ['UB1', 'UB2']
        loaders = [ub1_valid_loader, ub2_valid_loader]

        if option.is_train:
            trainer.train(train_loader, val_loaders=loaders, val_types=valid_types)
        else:
            for val_type, val_loader in zip(valid_types, loaders):
                trainer._validate(val_loader, step=0, valid_type=val_type)






if __name__ == '__main__': main()
