from task_adaptation.data import caltech
from task_adaptation.data import cifar
from task_adaptation.data import clevr
from task_adaptation.data import diabetic_retinopathy
from task_adaptation.data import dmlab
from task_adaptation.data import dsprites
from task_adaptation.data import dtd
from task_adaptation.data import eurosat
from task_adaptation.data import kitti
from task_adaptation.data import oxford_flowers102
from task_adaptation.data import oxford_iiit_pet
from task_adaptation.data import patch_camelyon
from task_adaptation.data import resisc45
from task_adaptation.data import smallnorb
from task_adaptation.data import sun397
from task_adaptation.data import svhn
from task_adaptation.data.base import compose_preprocess_fn
from task_adaptation.registry import Registry
import os
import os.path as osp
from PIL import Image

dataset_config = [
    ['caltech101', dict()],
    ['cifar', dict(num_classes=100)],
    ['dtd', dict()],
    ['oxford_flowers102', dict()],
    ['oxford_iiit_pet', dict()],
    ['patch_camelyon', dict()],
    ['sun397', dict()],
    ['svhn', dict()],
    ['resisc45', dict()],
    ['eurosat', dict()],
    ['dmlab', dict()],
    ['kitti', dict(task='closest_vehicle_distance')],
    ['smallnorb', dict(predicted_attribute='label_azimuth',dataset_postfix='azi')],
    ['smallnorb', dict(predicted_attribute='label_elevation',dataset_postfix='ele')],
    ['dsprites', dict(predicted_attribute='label_x_position',num_classes=16,dataset_postfix='loc')],
    ['dsprites', dict(predicted_attribute='label_orientation',num_classes=16,dataset_postfix='ori')],
    ['clevr', dict(task='closest_object_distance',dataset_postfix='dist')],
    ['clevr', dict(task='count_all',dataset_postfix='count')],
    ['diabetic_retinopathy', dict(config='btgraham-300')],
]

import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

data_root = osp.expanduser('../vtab-1k')

for dataset_name, dataset_params in dataset_config:
    dataset_postfix = dataset_params.pop('dataset_postfix', None)
    data_cls = Registry.lookup(f'data.{dataset_name}')(**dataset_params)
    
    if dataset_postfix is not None:
        dataset_name = dataset_name + '_' + dataset_postfix
    
    os.makedirs(f'{data_root}/{dataset_name}', exist_ok=True)
    os.makedirs(f'{data_root}/{dataset_name}/images', exist_ok=True)
    
    print(f'{dataset_name} started.')
    for split_name in ['train800', 'val200', 'test', 'train800val200']:
        data = data_cls._get_dataset_split(split_name=split_name, shuffle_files=False)
        base_preprocess_fn = compose_preprocess_fn(data_cls._image_decoder, data_cls._base_preprocess_fn)
        data = data.map(base_preprocess_fn, data_cls._num_preprocessing_threads)
        
        os.makedirs(f'{data_root}/{dataset_name}/images/{split_name}', exist_ok=True)
        
        with open(f'{data_root}/{dataset_name}/{split_name}.txt', 'w') as f:
            for i, item in enumerate(data):
                image_path = f'images/{split_name}/{i:06d}.jpg'
                label = item['label'].numpy().item()
                f.write(f'{image_path} {label}\n')
                
                image = item['image'].numpy()
                image = Image.fromarray(image)
                image.save(f'{data_root}/{dataset_name}/{image_path}')
    print(f'{dataset_name} is done.')
