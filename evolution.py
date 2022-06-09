import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from lib.datasets import build_dataset
from lib import utils
from supernet_engine_prompt import evaluate
import argparse
import os
import yaml
from lib.config import cfg, update_config_from_file

from mmcv.runner import get_dist_info, init_dist

import model as models
from timm.models import load_checkpoint

import sys


def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), list(cand_tuple[2 * depth + 1: 3 * depth + 1]), list(cand_tuple[3 * depth + 1: ])

class EvolutionSearcher(object):

    def __init__(self, args, device, model, model_without_ddp, choices, val_loader, test_loader, output_dir):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits 
        self.min_parameters_limits = args.min_param_limits 
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.s_prob =args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices

    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        
        # import pdb;pdb.set_trace()
        depth,visual_prompt_dim, lora_dim, adapter_dim,prefix_dim = decode_cand_tuple(cand)
        sampled_config = {}
        sampled_config['visual_prompt_dim'] = visual_prompt_dim
        sampled_config['lora_dim'] = lora_dim
        sampled_config['adapter_dim'] = adapter_dim
        sampled_config['prefix_dim'] = prefix_dim
        
        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        info['params'] =  n_parameters / 10.**6 

        if info['params'] > self.parameters_limits:
            print('parameters limit exceed')
            sys.stdout.flush()
            return False

        if info['params'] < self.min_parameters_limits:
            print('under minimum parameters limit')
            return False

        print("rank:", utils.get_rank(), cand, info['params'])
        # import pdb;pdb.set_trace()
        eval_stats = evaluate(self.val_loader, self.model, self.device, amp=self.args.amp, mode='retrain', retrain_config=sampled_config)
        # test_stats = evaluate(self.test_loader, self.model, self.device, amp=self.args.amp, mode='retrain', retrain_config=sampled_config)

        info['acc'] = eval_stats['acc1']
        # info['test_acc'] = test_stats['acc1']

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):

        cand_tuple = list()
        dimensions = ['visual_prompt_dim','lora_dim','adapter_dim','visual_prompt_depth','lora_depth','adapter_depth']
        depth = self.choices['depth']
        visual_prompt_depth = random.choice(self.choices['visual_prompt_depth'])
        lora_depth = random.choice(self.choices['lora_depth'])
        adapter_depth = random.choice(self.choices['adapter_depth'])
        cand_tuple.append(depth)

        cand_tuple += ([random.choice(self.choices['visual_prompt_dim']) for _ in range(visual_prompt_depth)] + [0] * (depth - visual_prompt_depth))
        cand_tuple += ([random.choice(self.choices['lora_dim']) for _ in range(lora_depth)] + [0] * (depth - lora_depth))
        cand_tuple += ([random.choice(self.choices['adapter_dim']) for _ in range(adapter_depth)] + [0] * (depth - adapter_depth))

        cand_tuple += [0]*12

        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth,visual_prompt_dim, lora_dim, adapter_dim, prefix_dim = decode_cand_tuple(cand)
            random_s = random.random()
            # visual_prompt_dim
            if random_s < m_prob:
                random_s = random.random()
                cur_depth = depth
                for idx,i in enumerate(visual_prompt_dim):
                    if i == 0:
                        cur_depth = idx
                        break
                if random_s < s_prob:
                    new_depth = random.choice(self.choices['visual_prompt_depth'])
                else:
                    new_depth = cur_depth

                if new_depth > cur_depth:
                    visual_prompt_dim = visual_prompt_dim[:cur_depth] + [random.choice(self.choices['visual_prompt_dim']) for _ in range(new_depth - cur_depth)] + [0]*(depth-new_depth)
                else:
                    visual_prompt_dim = visual_prompt_dim[:new_depth] + [0] * (depth - new_depth)

            # lora_dim
            random_s = random.random()
            if random_s < m_prob:
                random_s = random.random()
                cur_depth = depth
                for idx,i in enumerate(lora_dim):
                    if i == 0:
                        cur_depth = idx
                        break
                if random_s < s_prob:
                    new_depth = random.choice(self.choices['lora_depth'])
                else:
                    new_depth = cur_depth

                if new_depth > cur_depth:
                    lora_dim = visual_prompt_dim[:cur_depth] + [random.choice(self.choices['lora_dim']) for _ in range(new_depth - cur_depth)] + [0]*(depth-new_depth)
                else:
                    lora_dim = visual_prompt_dim[:new_depth] + [0] * (depth - new_depth)

            # adapter_dim
            random_s = random.random()
            if random_s < m_prob:
                random_s = random.random()
                cur_depth = depth
                for idx,i in enumerate(adapter_dim):
                    if i == 0:
                        cur_depth = idx
                        break
                if random_s < s_prob:
                    new_depth = random.choice(self.choices['adapter_depth'])
                else:
                    new_depth = cur_depth

                if new_depth > cur_depth:
                    adapter_dim = adapter_dim[:cur_depth] + [random.choice(self.choices['adapter_dim']) for _ in range(new_depth - cur_depth)] + [0]*(depth-new_depth)
                else:
                    adapter_dim = adapter_dim[:new_depth] + [0] * (depth - new_depth)

            result_cand = [depth]+visual_prompt_dim + lora_dim + adapter_dim + [0]*12
            
            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            cand_1 = list(random.choice(self.keep_top_k[k]))
            cand_2 = list(random.choice(self.keep_top_k[k]))
            p1 = decode_cand_tuple(cand_1)
            p2 = decode_cand_tuple(cand_2)

            final = list(random.choice([i, j]) for i, j in zip(p1[1:], p2[1:]))
            final = sum(final, [])
            final = [p1[0]] + final
            return tuple(final)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))


        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
            
            #updata top10
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            #updata top50
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 val acc = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['params']))   
                sys.stdout.flush()            
                # print('No.{} {} Top-1 val acc = {}, Top-1 test acc = {}, params = {}'.format(
                #     i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['test_acc'], self.vis_dict[cand]['params']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=0.64) 
    parser.add_argument('--min-param-limits', type=float, default=0)#default=18) 
    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # custom model argument
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    # parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'EVO_IMNET'],
    #                     type=str, help='Image Net dataset path')
    parser.add_argument('--data-set', default='IMNET', type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    parser.add_argument('--no_aug', action='store_true')


    parser.add_argument('--mode', type=str, default='search', choices=['super', 'search','retrain'], help='mode of AutoFormer')


    parser.add_argument('--few-shot-seed', type=int, default=0)
    parser.add_argument('--few-shot-shot', type=int, default=2)


    parser.add_argument('--IS_UNIPELT', action='store_true')
    parser.add_argument('--inception',action='store_true')
    parser.add_argument('--direct_resize',action='store_true')

    parser.add_argument('--IS_not_position_VPT',action='store_true')
    
    return parser

def main(args):
    update_config_from_file(args.cfg)

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        init_dist(launcher=args.launcher)
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # save config for later experiments
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
        f.write(args_text)
    # fix the seed for reproducibility

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher

    dataset_val, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_test, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(2 * args.batch_size),
        sampler=sampler_test, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    print(f"Creating SuperVisionTransformer")
    print(cfg)
    model = models.__dict__[cfg.MODEL_NAME](   
                                                img_size=args.input_size,
                                                drop_rate=args.drop,
                                                drop_path_rate=args.drop_path,
                                                super_prompt_tuning_dim=cfg.SUPERNET.VISUAL_PROMPT_DIM,super_LoRA_dim=cfg.SUPERNET.LORA_DIM,super_adapter_dim=cfg.SUPERNET.ADAPTER_DIM
                                                )

    if args.nb_classes != model.head.weight.shape[0]:
        model.reset_classifier(args.nb_classes)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # if args.nb_classes != model_without_ddp.head.weight.shape[0]:
    #     model_without_ddp.reset_classifier(args.nb_classes)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.resume:
        # import pdb;pdb.set_trace()
        incompatible_keys = load_checkpoint(model_without_ddp, args.resume,strict=False)
        print(incompatible_keys)


    choices = {'depth': cfg.SUPERNET.DEPTH,
               'super_prompt_tuning_dim':cfg.SUPERNET.VISUAL_PROMPT_DIM,
               'super_LoRA_dim':cfg.SUPERNET.LORA_DIM,
               'super_adapter_dim':cfg.SUPERNET.ADAPTER_DIM,
               'super_prefix_dim':cfg.SUPERNET.PREFIX_DIM,
               'visual_prompt_dim':cfg.SEARCH_SPACE.VISUAL_PROMPT_DIM,
               'lora_dim':cfg.SEARCH_SPACE.LORA_DIM,
               'adapter_dim':cfg.SEARCH_SPACE.ADAPTER_DIM,
               'prefix_dim':cfg.SEARCH_SPACE.PREFIX_DIM,
               'visual_prompt_depth':cfg.SEARCH_SPACE.VISUAL_PROMPT_DEPTH,
               'lora_depth':cfg.SEARCH_SPACE.LORA_DEPTH,
               'adapter_depth':cfg.SEARCH_SPACE.ADAPTER_DEPTH,
               'prefix_depth':cfg.SEARCH_SPACE.PREFIX_DEPTH,
               }


    t = time.time()
    searcher = EvolutionSearcher(args, device, model, model_without_ddp, choices, data_loader_val, data_loader_test, args.output_dir)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoFormer evolution search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
