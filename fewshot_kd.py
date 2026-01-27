#Modified from https://github.com/zju-vipa/Fast-Datafree
import argparse
from math import gamma
import os
import random
import shutil
import warnings
import math 

import registry
import inversion

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from typing import List, Optional, Tuple 
from batchsampler import BalancedClassBatchSampler

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

parser.add_argument('--method', required=True, choices=['zskt', 'dfad', 'dafl', 'deepinv', 'dfq',
                                                        'cmi', 'fast', 'fast_meta','demo1','demo2',
                                                        'demo3','PGDI'])
parser.add_argument('--adv', default=0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/synthesis', type=str)

parser.add_argument('--cr', default=1, type=float, help='scaling factor for contrastive model inversion')
parser.add_argument('--cr_T', default=0.5, type=float, help='temperature for contrastive model inversion')
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

parser.add_argument('--lr_g', default=1e-3, type=float, help='initial learning rate for generator')
parser.add_argument('--lr_z', default=1e-3, type=float, help='initial learning rate for latent code')
parser.add_argument('--g_steps', default=10, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--reset_l0', default=0, type=int,
                    help='reset l0 in the generator during training')
parser.add_argument('--reset_bn', default=0, type=int,
                    help='reset bn layers during training')
parser.add_argument('--bn_mmt', default=0, type=float,
                    help='momentum when fitting batchnorm statistics')
parser.add_argument('--is_maml', default=1, type=int,
                    help='meta gradient: is maml or reptile')

# Basic
parser.add_argument('--data_root', default='./data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')
parser.add_argument('--warmup', default=0, type=int, metavar='N',
                    help='which epoch to start kd')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--logits_guidance_w', default=0, type=float,
                    help='# Logits指导损失的权重 (默认为0，即禁用)')
parser.add_argument('--logits_loss_type', default='kl_softmax', type=str,
                    help='# Logits指导损失的类型')

parser.add_argument('--guidance_loss_w',    default=1, type=float, help='few-shot 特征指导权重')
parser.add_argument('--guidance_loss_type', default='l2_min', type=str, help='特征指导损失类型 (l2_min|l2_mean)')
parser.add_argument('--teacher_feature_source', default='bn_hooks', type=str, help='教师模型中提取特征指导的来源层')

parser.add_argument('--fewshot_n_per_class', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--real_fewshot_per_step', default=0, type=int,
                    help='每个蒸馏步骤中与合成批次混合的真实少样本图片数量。设置为0则禁用。')

parser.add_argument('--use_all_fewshot_per_step', action='store_true',
                    help='如果设置，则在每个蒸馏步骤中混合全部真实小样本图片，而不是按批次采样。')

parser.add_argument('--loss_ema_momentum', type=float, default=0.9,
                    help='用于平滑损失的指数移动平均动量 (mu)。范围: 0.0 到 1.0。 (默认: 0.9)')

parser.add_argument('--real_data_loss_weight', type=float, default=0.9,
                    help='课程学习法中 alpha 的初始值 (训练开始时)')
parser.add_argument('--alpha_schedule', type=str, default='cosine', choices=['none', 'linear', 'cosine'],
                    help='alpha 的动态调度策略 (none: 使用固定的 real_data_loss_weight)')
parser.add_argument('--alpha_initial', type=float, default=0.9,
                    help='课程学习法中 alpha 的初始值 (训练开始时)')

parser.add_argument('--real_fewshot_schedule', type=str, default='cosine', choices=['none', 'linear', 'cosine'],
                    help='每个蒸馏步骤中混合的真实图片数量的动态调度策略')
parser.add_argument('--real_fewshot_initial', type=int, default=64,
                    help='课程学习法中每个step使用的真实图片初始数量')

parser.add_argument('--mixed_distill_stop_epoch', type=int, default=500,
                    help='混合蒸馏第一阶段结束的 epoch。在此之后将进入第二阶段。')

parser.add_argument('--alpha_final_stage', type=float, default=0.05,
                    help='第二阶段中 alpha 的固定值 (一个很小的值)')

parser.add_argument('--real_fewshot_final_stage', type=int, default=4,
                    help='第二阶段中每个step使用的真实图片的固定数量 (一个很小的数目)')

best_acc1 = 0
time_cost = 0

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global time_cost
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = inversion.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s-%s%s'%(args.rank, args.dataset, args.teacher, args.student, args.log_tag) if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = inversion.utils.logger.get_logger(log_name, output='checkpoints/inversion-%s/log-%s-%s-%s%s.txt'%(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    if args.rank<=0:
        for k, v in inversion.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = inversion.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model

    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = normalizer = inversion.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    teacher_checkpoint_path = 'checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.teacher)
    checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu')

    state_dict_to_load = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict_to_load = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint: 
            state_dict_to_load = checkpoint['model_state_dict']
        elif 'model' in checkpoint: 
            state_dict_to_load = checkpoint['model']
        else:
            print(f"Warning: Checkpoint is a dictionary, but the 'state_dict', 'model_state_dict', or 'model' key was not found.")
            state_dict_to_load = checkpoint 
    else:
        state_dict_to_load = checkpoint

    if state_dict_to_load is not None:
        is_dataparallel = all(key.startswith('module.') for key in state_dict_to_load.keys())
        if is_dataparallel:
            state_dict_to_load = {k.replace('module.', ''): v for k, v in state_dict_to_load.items()}        
        try:
            teacher.load_state_dict(state_dict_to_load, strict=True)
        except RuntimeError as e:
            print(f"Error loading teacher model {args.teacher} weights (strict=True): {e}")
    else:
        print(f"Error: Failed to extract a valid state_dict from checkpoint file {teacher_checkpoint_path}.")
        
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = inversion.criterions.KLDiv(T=args.T)
    ############################################
    # Setup the data-free synthesizer
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.method=='deepinv':
        synthesizer = inversion.synthesis.DeepInvSyntheiszer(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, tv=0.0, l2=0.0,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
        nz = 512 if args.method=='dafl' else 256
        generator = inversion.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.method=='dfad' else inversion.criterions.KLDiv()
        synthesizer = inversion.synthesis.GenerativeSynthesizer(
                 teacher=teacher, student=teacher, generator=generator, nz=nz, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method=='cmi':
        nz = 256
        generator = inversion.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        feature_layers = None # use outputs from all conv layers
        if args.teacher=='resnet34': # use block outputs
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = inversion.synthesis.CMISynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), feature_reuse=False,
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method=='fast':
        nz = 256
        generator = inversion.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = inversion.synthesis.FastSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                 save_dir=args.save_dir, device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml)
    elif args.method=='fast_meta':
        nz = 256
        generator = inversion.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = inversion.synthesis.FastMetaSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                 save_dir=args.save_dir, device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml)   
    elif args.method=='PGDI':
        nz = 256
        generator = inversion.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = inversion.synthesis.PGDISynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                 save_dir=args.save_dir, device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0_ep=args.reset_l0, reset_bn_ep=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml,
                 guidance_loss_type=args.guidance_loss_type,
                 guidance_loss_w=args.guidance_loss_w,
                 logits_guidance_w=args.logits_guidance_w,logits_loss_type=args.logits_loss_type,
                 teacher_feature_source=args.teacher_feature_source,
                 fewshot_n_per_class=args.fewshot_n_per_class)
    else: raise NotImplementedError
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    teacher.eval()
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)


    real_fewshot_images_for_distillation = None
    real_fewshot_labels_for_distillation = None 

    if hasattr(synthesizer, 'get_fewshot_info_for_feature_guidance') and \
    args.real_fewshot_per_step > 0:       
        _, fs_images, fs_labels = synthesizer.get_fewshot_info_for_feature_guidance() 

        if fs_images is not None and fs_images.size(0) > 0:
            real_fewshot_images_for_distillation = fs_images.cpu().detach()
            real_fewshot_labels_for_distillation = fs_labels.cpu().detach() 
        else:
            print("Failed to retrieve real few-shot data for mixed distillation, or the number of samples is zero.")
    else:
        print(f"Current method {args.method} may not support or is not configured to get few-shot data.")

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        args.current_epoch=epoch
        for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            vis_results, cost = synthesizer.synthesize() # g_steps
            time_cost += cost
            if epoch >= args.warmup:
                train( synthesizer, [student, teacher], criterion, optimizer, args,
                                real_fewshot_images_for_distillation, real_fewshot_labels_for_distillation)# kd_steps
        for vis_name, vis_image in vis_results.items():
            inversion.utils.save_image_batch( vis_image, 'checkpoints/inversion-%s/%s%s.png'%(args.method, vis_name, args.log_tag) )

        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr']))

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/inversion-%s/%s-%s-%s.pth'%(args.method, args.dataset, args.teacher, args.student)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)

        if epoch >= args.warmup:
            scheduler.step()

    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)
        args.logger.info("Generation Cost: %1.3f" % (time_cost/3600.) )
        write_results_to_file(args, best_acc1)
    
    shutil.rmtree(args.save_dir)


def write_results_to_file(args, best_acc1):
    file_path = 'results.txt'
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            result_line = (
                f"fewshot_n_per_class: {args.fewshot_n_per_class}, "
                f"alpha_initial: {args.alpha_initial}, "
                f"alpha_final_stage: {args.alpha_final_stage}, "
                f"real_fewshot_initial: {args.real_fewshot_initial}, "
                f"real_fewshot_final_stage: {args.real_fewshot_final_stage}, "
                f"mixed_distill_stop_epoch: {args.mixed_distill_stop_epoch}, "
                f"best_acc1: {best_acc1:.4f}\n"  
            )
            f.write(result_line)
        print(f"Results have been successfully written to: {os.path.abspath(file_path)}")

    except IOError as e:
        print(f"Error writing to file: {e}")


def train(synthesizer, model, criterion, optimizer, args, real_fewshot_images=None, real_fewshot_labels=None): # 添加 real_fewshot_images 参数
    global time_cost # 
    
    loss_metric_synth = inversion.metrics.RunningLoss(inversion.criterions.KLDiv(reduction='sum'))
    loss_metric_real = inversion.metrics.RunningLoss(inversion.criterions.KLDiv(reduction='sum'))
    acc_metric = inversion.metrics.TopkAccuracy(topk=(1,5))
    
    student, teacher = model
    student.train()
    teacher.eval()
    
    real_fs_iter = None
    current_real_fs_batch_size = 0
    
    current_real_fs_batch_size = 0
    if args.current_epoch < args.mixed_distill_stop_epoch:
        if args.real_fewshot_schedule != 'none' and args.real_fewshot_per_step > 0:
            progress = args.current_epoch / args.mixed_distill_stop_epoch
            initial = args.real_fewshot_initial
            final = args.real_fewshot_final_stage 
            
            if args.real_fewshot_schedule == 'linear':
                current_real_fs_batch_size = int(initial - (initial - final) * progress)
            else: 
                current_real_fs_batch_size = int(final + 0.5 * (initial - final) * (1 + math.cos(math.pi * progress)))
        else:
            current_real_fs_batch_size = args.real_fewshot_initial
    else:
        current_real_fs_batch_size = args.real_fewshot_final_stage
    
    current_real_fs_batch_size = max(0, min(current_real_fs_batch_size, len(real_fewshot_images)))

    if current_real_fs_batch_size > 0 and real_fewshot_labels is not None:
        real_fs_dataset = torch.utils.data.TensorDataset(real_fewshot_images)
        balanced_sampler = BalancedClassBatchSampler(labels=real_fewshot_labels, batch_size=current_real_fs_batch_size)
        real_fs_dataloader = torch.utils.data.DataLoader(
            real_fs_dataset,
            batch_sampler=balanced_sampler,
            num_workers=0,
            pin_memory=False,
        )
        real_fs_iter = iter(real_fs_dataloader)
    ema_loss_synth = 0.0
    ema_loss_real = 0.0
    is_ema_initialized = False

    for i in range(args.kd_steps):
        synthetic_images_batch = None
        if args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
            sampled_data, cost = synthesizer.sample()
            time_cost += cost
            synthetic_images_batch = sampled_data
        else:
            synthetic_images_batch = synthesizer.sample()

        if synthetic_images_batch is None or synthetic_images_batch.size(0) == 0:
            args.logger.warning(f"Step {i}: Synthesizer did not return valid images, skipping this distillation step.")
            continue
        
        num_synthetic = synthetic_images_batch.size(0)
        current_real_fs_to_mix = None
        if args.use_all_fewshot_per_step and real_fewshot_images is not None:
            current_real_fs_to_mix = real_fewshot_images
        elif real_fs_iter is not None:
            try:
                current_real_fs_to_mix = next(real_fs_iter)[0]
            except StopIteration:
                real_fs_iter = iter(real_fs_dataloader)
                current_real_fs_to_mix = next(real_fs_iter)[0]
        if current_real_fs_to_mix is not None and current_real_fs_to_mix.size(0) > 0:
            if args.gpu is not None:
                synthetic_images_batch = synthetic_images_batch.cuda(args.gpu, non_blocking=True)
                current_real_fs_to_mix = current_real_fs_to_mix.cuda(args.gpu, non_blocking=True)
            else: 
                synthetic_images_batch = synthetic_images_batch.cpu()
                current_real_fs_to_mix = current_real_fs_to_mix.cpu()
            if synthetic_images_batch.shape[1:] != current_real_fs_to_mix.shape[1:]:
                args.logger.error(f"Shape mismatch! Synthetic images: {synthetic_images_batch.shape}, Real few-shot: {current_real_fs_to_mix.shape}. Skipping mixing.")
                combined_images_batch = synthetic_images_batch 
            else:
                combined_images_batch = torch.cat((synthetic_images_batch, current_real_fs_to_mix), dim=0)
        else:
            if args.gpu is not None:
                synthetic_images_batch = synthetic_images_batch.cuda(args.gpu, non_blocking=True)
            combined_images_batch = synthetic_images_batch

        if combined_images_batch is None or combined_images_batch.size(0) == 0:
            continue
        
        with args.autocast(): 
            with torch.no_grad():
                t_out, _ = teacher(combined_images_batch, return_features=True)
            s_out = student(combined_images_batch.detach()) 
            t_out_synth, s_out_synth = t_out[:num_synthetic], s_out[:num_synthetic]
            
            loss_synthetic = criterion(s_out_synth, t_out_synth.detach())
            loss_real = 0.0
            if combined_images_batch.size(0) > num_synthetic:
                t_out_real, s_out_real = t_out[num_synthetic:], s_out[num_synthetic:]
                loss_real = criterion(s_out_real, t_out_real.detach())
                
            alpha = args.real_data_loss_weight 
            if args.current_epoch < args.mixed_distill_stop_epoch:
                if args.alpha_schedule != 'none':
                    progress = args.current_epoch / args.mixed_distill_stop_epoch
                    initial = args.alpha_initial
                    final = args.alpha_final_stage 
                    
                    if args.alpha_schedule == 'linear':
                        alpha = initial - (initial - final) * progress
                    else: 
                        alpha = final + 0.5 * (initial - final) * (1 + math.cos(math.pi * progress))
                else:
                    alpha = args.alpha_initial
            else:
                alpha = args.alpha_final_stage                        
            alpha = max(0.0, min(1.0, alpha)) 
            

            loss_total = alpha * loss_real + (1 - alpha) * loss_synthetic

        optimizer.zero_grad()
        if args.fp16:           
            args.scaler.scale(loss_total).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
        else:
            loss_total.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1]) 
        loss_metric_synth.update(s_out_synth, t_out_synth)
        if combined_images_batch.size(0) > num_synthetic:
            loss_metric_real.update(s_out_real, t_out_real)

        if (args.print_freq > 0 and i % args.print_freq == 0) or \
            (args.print_freq == -1 and i % 10 == 0 and args.current_epoch >= 150):
            (train_acc1, train_acc5) = acc_metric.get_results()
            train_loss_synth = loss_metric_synth.get_results() 
            train_loss_real = loss_metric_real.get_results()
            args.logger.info(
                f'[Train Mix] Epoch={args.current_epoch} Iter={i}/{args.kd_steps}, '
                f'Acc@1={train_acc1:.4f}, TotalLoss={loss_total.item():.4f}, '
                f'L_synth={train_loss_synth:.4f}, L_real={train_loss_real:.4f}, '
                f'alpha={alpha:.3f}, Lr={optimizer.param_groups[0]["lr"]:.4f}'
            )
            loss_metric_synth.reset()
            loss_metric_real.reset()
            acc_metric.reset()
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()