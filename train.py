#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import *
from models.csnet import NetWork
from dataloader import SegDataLoader
from loss import *
from configs import config_factory
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import cv2

import os
import logging
import time
import datetime
import argparse


cfg = config_factory['cfg']
if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
if not os.path.isdir(cfg.view_path): os.makedirs(cfg.view_path)
torch.backends.cudnn.benchmark = True

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()


def train(verbose=True, **kwargs):
    args = kwargs['args']
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:{}'.format(cfg.port),
                world_size = torch.cuda.device_count(),
                rank = args.local_rank
                )
    setup_logger(cfg.respth)
    logger = logging.getLogger()
    mIOU=0
    ## dataset
    ds = SegDataLoader(cfg, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = cfg.ims_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = cfg.n_workers,
                    pin_memory = True,
                    drop_last = True)

    ## model
    print('model')
    net = NetWork(cfg)
    print('cuda')
    net.cuda()
    print('Done')

    ## resume
    if cfg.resume:
        print("=> loading checkpoint '{}'".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        if '.tar' in  cfg.resume:
            net.load_state_dict(checkpoint['model'])

            print('Pth.Tar Load model from {}'.format(cfg.resume))
        else:
            net.load_state_dict(checkpoint)
            print('Pth Load model from {}'.format(cfg.resume))
        print('pretrained model loaded')
        del checkpoint

    it_start = 0
    n_epoch = 0
    net = nn.parallel.DistributedDataParallel(net,
           device_ids = [args.local_rank, ],
           output_device = args.local_rank
           )


    if hasattr(net, 'module'):
        bk_wd_params, bk_no_wd_params, wd_params, no_wd_params = net.module.get_params()
    else:
        bk_wd_params, bk_no_wd_params, wd_params, no_wd_params = net.get_params()

    params_list = [{'params': bk_wd_params, 'weight_decay': cfg.weight_decay, 'lr':1e-2*cfg.lr_start},
                {'params': bk_no_wd_params, 'weight_decay': 0, 'lr':2e-2*cfg.lr_start},
                {'params': wd_params, 'weight_decay': cfg.weight_decay, 'lr':cfg.lr_start},
                {'params': no_wd_params, 'weight_decay': 0, 'lr':2*cfg.lr_start}]

    optim  =torch.optim.SGD(params_list, lr = cfg.lr_start, momentum = cfg.momentum, weight_decay = cfg.weight_decay)
    scheduler_steplr = StepLR(optim, step_size=40, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)


    net.train()
    n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16

    criteria_seg = CE(ignore_lb=255, weight=None).cuda()


    ## train loop
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    # n_epoch = 0
    counter = 0
    #count for the epoch finished
    epochF = 0

    for it in range(it_start, cfg.total_iter):
        try:
            im, lb, dp, _ = next(diter)
            if not im.size()[0]==cfg.ims_per_gpu: continue
        except StopIteration:
            n_epoch += 1
            scheduler_warmup.step(n_epoch+1)
            sampler.set_epoch(n_epoch)
            diter = iter(dl)
            im, lb, dp, _ = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        dp = dp.cuda()

        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1).long()

        try:
            loss = torch.zeros([1]).cuda()
            optim.zero_grad()
            logits4, logits8, logits16, logits32 = net(im, dp)

            loss += criteria_seg(logits4, lb)
            loss += criteria_seg(logits8, lb)
            loss += criteria_seg(logits16, lb)
            loss += criteria_seg(logits32, lb)


            loss.backward()
            optim.step()
            scheduler_warmup.step(n_epoch+1)
        except RuntimeError as e:
            if 'out of memory' in e:
                print('| WARNING: run out of memory')
                if hasattr(troch.cuda, 'empty_cach'):
                    torch.cuda.empty_cache()
            else:
                raise e

        torch.cuda.empty_cache()
        loss_avg.append(loss.item())
        ## print training log message
        if it%cfg.msg_iter==0 and not it==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = []
            for pg in optim.param_groups:
                lr.append(pg['lr'])
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            msg = ', '.join([
                    'epoch:{epoch}'
                    'iter: {it}/{max_it}',
                    'bk_lr: {bk_lr:4f}',
                    'bk_nw_lr: {bk_nw_lr:4f}',
                    'de_lr: {de_lr:4f}',
                    'de_nw_lr: {de_nw_lr:4f}',
                    'loss: {loss:.4f}',
                    'time: {time:.4f}',
                ]).format(
                    epoch = n_epoch,
                    it = it,
                    max_it = cfg.total_iter,
                    bk_lr = lr[0],
                    bk_nw_lr = lr[1],
                    de_lr = lr[2],
                    de_nw_lr = lr[3],
                    loss = loss_avg,
                    time = t_intv,
                )

            logger.info(msg)
            loss_avg = []
            st = ed

        if n_epoch > epochF and n_epoch >= 20 and (n_epoch % 5 == 0):

            epochF = n_epoch

            save_pth = osp.join(cfg.respth, 'checkpoint.pth.tar')
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            if dist.get_rank()==0:
                stateF = {
                        'model': state,
                        'it': it,
                        'epoch': n_epoch,
                        'optimizer': optim.state_dict(),

                    }
                torch.save(stateF, save_pth)

            net.train()

        if (n_epoch + 1) == cfg.max_epoch:
            print('break')
            break

    if verbose:
        net.cpu()
        save_pth = osp.join(cfg.respth, 'model_final.pth.rar')
        state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        stateF = {
            'model': state,
            'mIOU': mIOU,
            'it': it,
            'epoch': n_epoch,
            'optimizer': optim.state_dict(),

            }
        torch.save(stateF, save_pth)
        if dist.get_rank()==0: torch.save(state, save_pth)
        logger.info('training done, model saved to: {}'.format(save_pth))

if __name__ == "__main__":
    args = parse_args()
    train(args=args)
