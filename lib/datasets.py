import os
import cv2
import json
import torch
import scipy
import scipy.io as sio
from skimage import io

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

class general_dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        if mode == 'super' and is_individual_prompt==False:
            train_list_path = os.path.join(self.dataset_root, 'train800.txt')
        elif mode == 'search' and is_individual_prompt==False:
            train_list_path = os.path.join(self.dataset_root, 'val200.txt')
        else:
            train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')

        if mode != 'search':
            test_list_path = os.path.join(self.dataset_root, 'test.txt')
        else:
            test_list_path = os.path.join(self.dataset_root, 'val200.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))

class general_dataset_few_shot(ImageFolder):
    def __init__(self, root, dataset,train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,shot=2,seed=0,**kwargs):
        self.dataset_root = root
        self.dataset = dataset.replace('-FS','')
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        
        if mode == 'super' and is_individual_prompt==False:
            train_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_'+str(shot)+'.seed_'+str(seed))
        elif mode == 'search' and is_individual_prompt==False:
            if 'imagenet' in root:
                train_list_path = os.path.join(self.dataset_root, 'annotations/unofficial_val_list_4_shot16seed0')
            else:
                train_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
        else:
            if 'imagenet' in root and self.dataset != 'imagenet':
                train_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
            else:
                train_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_'+str(shot)+'.seed_'+str(seed))

        if mode == 'search':
            test_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_1.seed_0')           
        elif 'imagenet' in root:
            test_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
        else:
            test_list_path = os.path.join(self.dataset_root, 'annotations/test_meta.list')


        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in root or ('imagenet' in root and 'imagenet' != self.dataset):
                        self.samples.append((os.path.join(root,img_name), label))
                    elif 'imagenet' == self.dataset:
                        self.samples.append((os.path.join(root+'/train',img_name), label))
                    else:
                        self.samples.append((os.path.join(root+'/images',img_name), label))
                    
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in root or ('imagenet' in root and 'imagenet' != self.dataset):
                        self.samples.append((os.path.join(root,img_name), label))
                    elif 'imagenet' == self.dataset:
                        if mode == 'search':
                            self.samples.append((os.path.join(root+'/train',img_name), label))
                        else:
                            self.samples.append((os.path.join(root+'/val',img_name), label))
                    else:
                        self.samples.append((os.path.join(root+'/images',img_name), label))


def build_dataset(is_train, args, folder_name=None,is_individual_prompt=False):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CARS':
        dataset = Cars196(args.data_path, train=is_train, transform=transform)
        nb_classes = 196
    elif args.data_set == 'PETS':
        dataset = Pets(args.data_path, train=is_train, transform=transform)
        nb_classes = 37
    elif args.data_set == 'FLOWERS':
        dataset = Flowers(args.data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == 'IMNET':
        dataset = ImageNet(args.data_path, train=is_train, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'EVO_IMNET':
        root = os.path.join(args.data_path, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'clevr_count':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 8
    elif args.data_set == 'diabetic_retinopathy':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 5
    elif args.data_set == 'dsprites_loc':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 16
    elif args.data_set == 'dtd':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 47
    elif args.data_set == 'kitti':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 4
    elif args.data_set == 'oxford_pet':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 37
    elif args.data_set == 'resisc45':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 45
    elif args.data_set == 'smallnorb_ele':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 9
    elif args.data_set == 'svhn':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 10
    elif args.data_set == 'cifar100':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 100
    elif args.data_set == 'clevr_dist':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 6
    elif args.data_set == 'caltech101':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 102
    elif args.data_set == 'dmlab':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 6
    elif args.data_set == 'dsprites_ori':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 16
    elif args.data_set == 'eurosat':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 10
    elif args.data_set == 'oxford_flowers102':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 102
    elif args.data_set == 'patch_camelyon':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 2
    elif args.data_set == 'smallnorb_azi':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 18
    elif args.data_set == 'sun397':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt)
        nb_classes = 397
    elif '-FS' in args.data_set:
        dataset = general_dataset_few_shot(args.data_path, args.data_set,train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt,shot=args.few_shot_shot,seed=args.few_shot_seed)
        if 'stanford_cars' in args.data_set:
            nb_classes = 196
        elif 'oxford_flowers' in args.data_set:
            nb_classes = 102
        elif 'food-101' in args.data_set:
            nb_classes = 101
        elif 'oxford_pets'in args.data_set:
            nb_classes = 37
        elif 'fgvc_aircraft' in args.data_set:
            nb_classes = 100
        elif 'imagenet' in args.data_set:
            nb_classes = 1000

    return dataset, nb_classes

def build_transform(is_train, args):
    if not args.no_aug and is_train and args.mode != 'search':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        return transform

    t = []
    if args.direct_resize:
        size = args.input_size
    else:
        size = int((256 / 224) * args.input_size)

    t.append(
        transforms.Resize((size,size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.inception:
        t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
