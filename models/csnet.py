#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torchvision
from cc_attention import CrissCrossAttention
from modules import InPlaceABNSync as BatchNorm2d
from .resnet import Resnet101

def Up(x, shape):
    return F.interpolate(x, shape, mode='bilinear', align_corners=True)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = True)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.2)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FRM(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, dils=[1, 2, 5], *args, **kwargs):
        super(FRM, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(out_chan, out_chan, ks=3, dilation=dils[0], padding=dils[0])
        self.conv3 = ConvBNReLU(out_chan, out_chan, ks=3, dilation=dils[1], padding=dils[1])
        self.conv4 = ConvBNReLU(out_chan, out_chan, ks=3, dilation=dils[2], padding=dils[2])
        self.conv_out = ConvBNReLU(out_chan*4, out_chan, ks=1, padding=0)
        self.cca1 = CrissCrossAttention(out_chan)
        self.cca2 = CrissCrossAttention(out_chan)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2+feat1)
        feat4 = self.conv4(feat3+feat2+feat1)
        feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        feat_cca = self.cca2(self.cca1(feat))
        return feat + feat_cca

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.2)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class TransLayer(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(TransLayer, self).__init__()
        self.layer = ConvBNReLU(in_chan, out_chan, ks=3, padding=1)
    def forward(self, x):
        return self.layer(x)




class FineEnc(nn.Module):
    def __init__(self, low_chan=[2048, 1024, 512, 256], *args, **kwargs):
        super(FineEnc, self).__init__()
        self.conv_4 = TransLayer(low_chan[3], 256)
        self.conv_8 = TransLayer(low_chan[2], 256)
        self.conv_16 = TransLayer(low_chan[1], 256)
        self.conv_32 = TransLayer(low_chan[0], 512)

        self.frm_feat4 = FRM(256, 256)
        self.frm_feat8 = FRM(256, 256)
        self.frm_feat16 = FRM(512, 512)
        self.frm_feat32 = FRM(512, 256)

        self.conv_up1 = ConvBNReLU(512, 256, ks=1, padding=0)
        self.conv_up2 = ConvBNReLU(512, 512, ks=1, padding=0)
        self.conv_up3 = ConvBNReLU(1024, 512, ks=1, padding=0)

        self.init_weight()

    def forward(self, feat4, feat8, feat16, feat32):

        feat4_trans = self.frm_feat4(self.conv_4(feat4))
        feat8_trans = self.conv_8(feat8)

        up1 = self.frm_feat8(self.conv_up1(torch.cat([Up(feat4_trans, feat8_trans.size()[2:]), feat8_trans], 1)))

        feat16_trans = self.conv_16(feat16)
        up2 = self.frm_feat16(self.conv_up2(torch.cat([Up(up1, feat16_trans.size()[2:]), feat16_trans], 1)))

        feat32_trans = self.conv_32(feat32)
        up3 = self.conv_up3(torch.cat([Up(up2, feat32_trans.size()[2:]), feat32_trans], 1))

        feat32_trans = self.frm_feat32(up3)


        return feat4_trans, feat8_trans, feat16_trans, feat32_trans

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.2)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ChannelWeights(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelWeights, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weight()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.2)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AFM(nn.Module):
    def __init__(self, in_chan1, in_chan2, out_chan, n_classes, gen_feat=True):
        super(AFM, self).__init__()
        self.conv1_1 = ConvBNReLU(in_chan1, out_chan, ks=1, padding=0)
        self.gen_feat = gen_feat
        if gen_feat:
            self.conv1_2 = ConvBNReLU(2*out_chan, out_chan, ks=1, padding=0)

        self.conv2_1 = ConvBNReLU(in_chan2, out_chan, ks=1, padding=0)
        self.conv2_2 = ConvBNReLU(2*out_chan, out_chan, ks=1, padding=0)

        self.cw = ChannelWeights(2*out_chan)

        self.seg = nn.Sequential(nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False),
                                 nn.Conv2d(out_chan, n_classes, kernel_size=1, bias=False))

    def forward(self, prefeat, bkfeat):
        prefeat1 = Up(self.conv1_1(prefeat), bkfeat.size()[2:])
        bkfeat1 = self.conv2_1(bkfeat)

        feat_cat = torch.cat([prefeat1, bkfeat1], 1)
        c_w = self.cw(feat_cat)
        feat_cat = c_w*feat_cat + feat_cat

        sgfeat = self.conv2_2(feat_cat)
        logits = self.seg(sgfeat)

        if self.gen_feat:
            nxfeat = self.conv1_2(feat_cat)
            return nxfeat, logits
        return logits

class EncDec(nn.Module):
    def __init__(self, n_classes, low_chan=[2048, 1024, 512, 256], *args, **kwargs):
        super(EncDec, self).__init__()
        self.conv_4_low = TransLayer(low_chan[3], 128)

        self.f_co = TransLayer(low_chan[0], 256)

        self.conv_cat = nn.Sequential(
                ConvBNReLU(384, 256, ks=3, padding=1),
                ConvBNReLU(256, 256, ks=3, padding=1),
                )

        self.finenc = FineEnc(low_chan)

        self.coarse_feat = nn.Sequential(ConvBNReLU(256, 256, kernel_size=3),
                                    ConvBNReLU(256, 256, kernel_size=3))

        self.fuse32 = TransLayer(512, 256)


        self.fuse16 = TransLayer(512, 256)
        self.fuse8 = TransLayer(512, 256)
        self.fuse4 = TransLayer(512, 256)

        self.afm32 = AFM(256, 256, 256, n_classes)
        self.afm16 = AFM(256, 256, 256, n_classes)
        self.afm8 = AFM(256, 256, 256, n_classes)
        self.afm4 = AFM(256, 256, 256, n_classes, gen_feat=False)

        self.init_weight()

    def forward(self, feat4, feat8, feat16, feat32, evaluation=False):
        feat4_low = self.conv_4_low(feat4)
        co_feat = self.f_co(feat32)
        coarse_dec = self.conv_cat(torch.cat([feat4_low, Up(co_feat, feat4_low.size()[2:])], 1))

        coarse_feat = self.coarse_feat(coarse_dec)

        feat4_trans, feat8_trans, feat16_trans, feat32_trans = self.finenc(feat4, feat8, feat16, feat32)

        cat_feat32 = self.fuse32(torch.cat([co_feat, feat32_trans], 1))

        feat16_32 = self.fuse16(torch.cat([cat_feat32, feat16_trans], 1))
        feat8_16 = self.fuse8(torch.cat([feat8_trans, Up(feat16_32, feat8_trans.size()[2:])], 1))
        feat4_8 = self.fuse4(torch.cat([feat4_trans, Up(feat8_16, feat4_trans.size()[2:])], 1))

        nxfeat32, logits32 = self.afm32(coarse_feat, cat_feat32)
        nxfeat16, logits16 = self.afm16(nxfeat32, feat16_32)
        nxfeat8, logits8 = self.afm8(nxfeat16, feat8_16)
        logits4 = self.afm4(nxfeat8, feat4_8)

        return logits4, logits8, logits16, logits32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.2)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params = []
        non_wd_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params


class NetWork(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(NetWork, self).__init__()
        self.backbone = Resnet101(16)
        self.encdec = EncDec(cfg.n_classes, low_chan=[2048, 1024, 512, 256])


    def forward(self, x, depth, evaluation=False):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(torch.cat([x, depth], 1))
        logits4, logits8, logits16, logits32 = self.encdec(feat4, feat8, feat16, feat32)
        logits4 = Up(logits4, (H, W))
        logits8 = Up(logits8, (H, W))
        logits16 = Up(logits16, (H, W))
        logits32 = Up(logits32, (H, W))

        return logits4, logits8, logits16, logits32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.2)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        bk_wd_params, bk_no_wd_params = self.backbone.get_params()
        decoder_wd_params, decoder_no_wd_params = self.decoder.get_params()

        return bk_wd_params, bk_no_wd_params, decoder_wd_params, decoder_no_wd_params

