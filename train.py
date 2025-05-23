import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt 
import requests
from dotenv import load_dotenv
import shutil
from collections import defaultdict
import re


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset, config
from util.s3dis import S3DIS
from util.scannet_v2 import Scannetv2
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, poly_learning_rate, smooth_loss
from util.data_util import collate_fn, collate_fn_limit
from util import transform
from util.logger import get_logger

from functools import partial
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup
import torch_points_kernels as tp


# task night 26/3 : 

# Modify the code so that it records down the following metrics into following file ✅ 
# 1. mean Iou mean Acc across each validation  , if paramEpoch = 10, then will be 10 rows 
# 2. Iou and Acc across each each validation , if paramEpoch = 10 ,and val data = 4, then  will be 40 rows  ✅ 
# 3. mean Iou mean Acc across each training , if paramEpoch = 10 , then will have 10 rows ✅ 
# 4. Iou and Acc across each each training , if paramEpoch = 10 ,and training data = 10, will be large number like 2940..✅ 
# 5. Iou and Acc for each class (0-3) across each validation , , if paramEpoch = 10, will have 10 rows✅ 
 
# Ouput File Name :
# 1. validation_logs_5_mean.csv ✅ 
# 2. validation_logs_5.csv  ✅ 
# 3. training_logs_5_mean.csv ✅ 
# 4. training_logs_5.csv  ✅ 
# 5. validation_per_class.csv✅ 


load_dotenv()

telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")


train_loss = []
train_avg_loss = []
val_loss = []
val_avg_loss = []
train_acc = []
val_acc = []
epochs_count = []

train_mIou = []
train_mAcc = []
train_allAcc = []
train_Iou = []


val_mIou = []
val_mAcc = []
val_Iou = []

class_val_iou = []
class_val_acc = []


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_stratified_transformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    print("Len of train gpu :",args.train_gpu)
    if len(args.train_gpu) == 1:

        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        print("main worker !")
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # get model
    if args.arch == 'stratified_transformer':
        
        from model.stratified_transformer import Stratified

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
            args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    elif args.arch == 'swin3d_transformer':
        
        from model.swin3d_transformer import Swin

        args.patch_size = args.grid_size * args.patch_size
        args.window_sizes = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Swin(args.depths, args.channels, args.num_heads, \
            args.window_sizes, args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, \
            concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    
    # set loss func 
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    
    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            
            # Filter out classifier weights and other mismatched layers
            model_dict = model.state_dict()
            pretrained_dict = {}
            for k, v in checkpoint['state_dict'].items():
                # Skip classifier layer
                if 'classifier.3' in k:
                    if main_process():
                        logger.info(f"Skipping {k} due to classifier mismatch")
                    continue
                    
                # Skip items that don't exist in current model
                if k not in model_dict:
                    if main_process():
                        logger.info(f"Skipping {k} as it doesn't exist in current model")
                    continue
                    
                # Skip items with size mismatch
                if v.size() != model_dict[k].size():
                    if main_process():
                        logger.info(f"Skipping {k} due to size mismatch: {v.size()} vs {model_dict[k].size()}")
                    continue
                    
                # Add matched items
                pretrained_dict[k] = v
                
            # Update model with filtered weights
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            if main_process():
                logger.info(f"=> loaded {len(pretrained_dict)}/{len(model_dict)} layers from weight '{args.weight}'")
    else:
        logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 's3dis':
        train_transform = None
        if args.aug:
            jitter_sigma = args.get('jitter_sigma', 0.01)
            jitter_clip = args.get('jitter_clip', 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info("jitter_sigma: {}, jitter_clip: {}".format(jitter_sigma, jitter_clip))
            train_transform = transform.Compose([
                transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
                transform.RandomDropColor(color_augment=args.get('color_augment', 0.0))
            ])
        train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    elif args.data_name == 'scannetv2':
        train_transform = None
        if args.aug:
            if main_process():
                logger.info("use Augmentation")
            train_transform = transform.Compose([
                transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform.RandomDropColor(color_augment=args.get('color_augment', 0.0))
            ])
            
        train_split = args.get("train_split", "train")
        if main_process():
            logger.info("scannet. train_split: {}".format(train_split))

        train_data = Scannetv2(split=train_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None))

    val_transform = None
    if args.data_name == 's3dis':
        val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
    elif args.data_name == 'scannetv2':
        val_data = Scannetv2(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, \
            pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
    
    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
            warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == 'MultiStep':
        assert args.scheduler_update == 'epoch'
        milestones = [int(x) for x in args.milestones.split(",")] if hasattr(args, "milestones") else [int(args.epochs*0.6), int(args.epochs*0.8)]
        gamma = args.gamma if hasattr(args, 'gamma') else 0.1
        if main_process():
            logger.info("scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(args.scheduler_update, milestones, gamma))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    for epoch in range(args.start_epoch, args.epochs):
   
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))
            
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1
        
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        
        data_time.update(time.time() - end)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
    
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]
        
        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(feat, coord, offset, batch, neighbor_idx)
            assert output.shape[1] == args.classes
            if target.shape[-1] == 1:
                target = target[:, 0]  # for cls
            loss = criterion(output, target)
            
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          lr=lr,
                                                          accuracy=accuracy))
            train_loss.append(loss_meter.val)
            train_acc.append(accuracy)
            train_avg_loss.append(loss_meter.avg)
            epochs_count.append(epoch+1)
            
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    
    # debugging
    print(f"intersection_meter.sum: {intersection_meter.sum}, type: {type(intersection_meter.sum)}")
    print(f"target_meter.sum: {target_meter.sum}, type: {type(target_meter.sum)}")


    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    # sw part
    allAcc = intersection_meter.sum / (target_meter.sum + 1e-10) if isinstance(intersection_meter.sum, (int, float)) else sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    if main_process():
        train_mAcc.append(mAcc)
        train_mIou.append(mIoU)
        train_allAcc.append(allAcc)
        
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
    
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
    
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]
        
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        with torch.no_grad():
            output = model(feat, coord, offset, batch, neighbor_idx)
            loss = criterion(output, target)

        output = output.max(1)[1]
        
        print("Unique Target Classes:", np.unique(target.cpu().numpy()))  # 👈 Add this


        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
            
            val_loss.append(loss_meter.val)
            val_acc.append(accuracy)
            val_avg_loss.append(loss_meter.avg)
            
            
    print("In validation : ")
    print(f"Intersection Meter - Sum: {intersection_meter.sum}, Count: {intersection_meter.count}, Avg: {intersection_meter.avg}")
    print(f"Target Meter - Sum: {target_meter.sum}, Count: {target_meter.count}, Avg: {target_meter.avg}")

    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = np.nan_to_num(intersection_meter.sum / (target_meter.sum + 1e-10), nan=0.0).mean()

    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        
        val_mIou.append(mIoU)
        val_mAcc.append(mAcc)
        class_val_iou.append(iou_class)
        class_val_acc.append(accuracy_class)
                
        
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
            


        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


def sendTelegramNotification(msg):
    """
    Sends a message to your Telegram account.
    """
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": telegram_chat_id,
        "text": msg
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send message to Telegram: {e}")
        return None


def get_ss_scene_prefixes(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    scene_groups = defaultdict(list)

    for f in files:
        if f.startswith("scene"):
            match = re.match(r"(scene\d+_\d+)",f)
            if match:
                prefix = match.group(1)
                scene_groups[prefix].append(f)
            else:
                print(f"Warning: No match for file {f}")

    return scene_groups

def get_ori_scene_prefixes(ss_list):

    original_list = set()
    for f in ss_list:
        if f.startswith("scene"): # all looks like scene0152_15 , wan to look at last 2 number 
            prefix = f.split("_")[1]  # e.g. scene0151_03
            name = f[:-4]
            ori_name = name+prefix
            original_list.add(ori_name)
    
    return list(original_list)



def get_scene_prefixes(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    scene_groups = defaultdict(list)
    for f in files:
        if f.startswith("scene"):
            match = re.match(r"(scene\d+_\d+)", f)
            if match:
                prefix = match.group(1)
                scene_groups[prefix].append(f)
    return scene_groups

# forgive me i had to hardcode this
TRAIN_DIR = '/home/swlee/scannetdata/PointGroup/dataset/scannetv2/PointGroup/dataset/scannetv2/train'
VAL_DIR = '/home/swlee/scannetdata/PointGroup/dataset/scannetv2/PointGroup/dataset/scannetv2/val'
# line of hardcoded

def reset_folders():
    


    """Move all scene files from val/ back to train/."""
    val_scenes = get_scene_prefixes(VAL_DIR)
    for prefix, files in val_scenes.items():
        for f in files:
            shutil.move(os.path.join(VAL_DIR, f), os.path.join(TRAIN_DIR, f))

def apply_loocv(fold_idx, scene_prefixes):
  
    reset_folders()
    val_scene = scene_prefixes[fold_idx]
    print(f"[Fold {fold_idx + 1}] Moving scene '{val_scene}' to val set...")
    
    all_val_scene = []
    
    for i in range(5):
        
        scene = val_scene[:8] + str(i) + val_scene[9:]
        all_val_scene.append(scene)
    
    print(f"all_val_scene = {all_val_scene}")
    
    # Move this scene to val
    for val_scene in all_val_scene:
        for f in os.listdir(TRAIN_DIR):
            if f.startswith(val_scene):
                print(f"Moving {f} to val set")
                shutil.move(os.path.join(TRAIN_DIR, f), os.path.join(VAL_DIR, f))

    print(f"[Fold {fold_idx + 1}] Val scene = {val_scene}, remaining in train = {len(scene_prefixes) - 1}\n")
    
    for val_scene in all_val_scene:
        for f in os.listdir(VAL_DIR):
            if f.startswith(val_scene):
                print(f"Val set contains: {f}")
    
    input(f"Fold {fold_idx+1} Press Enter to continue...")

    
    
def save_results():
    # save and visualize training loss/acc/epoch ..

    
    # 1st
    df_train = pd.DataFrame({
         "Epochs" : epochs_count,
         "Training Loss": train_loss,
         "Training Accuracy": train_acc,
         "Training Avg Loss (so far)" : train_avg_loss

         
    })
    
    print(len(train_mIou),len(train_mAcc))
    df_train_mean = pd.DataFrame({
        #  "Epochs" : epochs_count,
         "mIou": train_mIou,
         "mAcc": train_mAcc,
         "allAcc" : train_allAcc

    })
    
    df_val = pd.DataFrame({
        "Validation Loss": val_loss,
        "Validation Accuracy":val_acc,
        "Validation Avg Loss" : val_avg_loss
    })
    
    df_val_mean = pd.DataFrame({
        
        #  "Epochs" : epochs_count,
         "mIou": val_mIou,
         "mAcc": val_mAcc,
        
    })
    
    df_class_val = pd.DataFrame({
        # "Epochs": epochs_count,
        "Class 0 Iou" : [i[0] for i in class_val_iou],
        "Class 0 Acc" :  [i[0] for i in class_val_acc],
        "Class 1 Iou" : [i[1] for i in class_val_iou],
        "Class 1 Acc" : [i[1] for i in class_val_acc],
        "Class 2 Iou" :[i[2] for i in class_val_iou] ,
        "Class 2 Acc": [i[2] for i in class_val_acc],
        "Class 3 Iou" : [i[3] for i in class_val_iou],
        "Class 3 Acc":[i[3] for i in class_val_acc]
    })
    
    
    training_logs = f"training_logs_{args.train_id}.csv" 
    training_mean_logs = f"training_logs_{args.train_id}_mean.csv"
    
    val_logs = f"validation_logs_{args.train_id}.csv"
    val_mean_logs = f"validation_logs_{args.train_id}_mean.csv"
    
    class_val_logs = f"validation_per_class_{args.train_id}.csv"
    
    
    # Define full file paths
    training_file_path = os.path.join(args.excel_folder, training_logs)
    training_mean_file_path = os.path.join(args.excel_folder, training_mean_logs)

    val_file_path = os.path.join(args.excel_folder, val_logs)
    val_mean_file_path = os.path.join(args.excel_folder, val_mean_logs)

    class_val_file_path = os.path.join(args.excel_folder, class_val_logs)

    # Save DataFrames to CSV files
    df_train.to_csv(training_file_path, index=False)
    print(f"✅ Training records saved to {training_file_path}")

    df_train_mean.to_csv(training_mean_file_path, index=False)
    print(f"✅ Training mean metrics saved to {training_mean_file_path}")

    df_val.to_csv(val_file_path, index=False)
    print(f"✅ Validation records saved to {val_file_path}")

    df_val_mean.to_csv(val_mean_file_path, index=False)
    print(f"✅ Validation mean metrics saved to {val_mean_file_path}")

    df_class_val.to_csv(class_val_file_path, index=False)
    print(f"✅ Validation per class saved to {class_val_file_path}")
    
    sendTelegramNotification(f"✅ Training Done , Result saved Successfully into {args.excel_folder}")
    


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
    save_results()
    
