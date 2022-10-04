import torch
from   torch import nn as nn

from utils import *

class ConvBnPrelu(nn.Module):
    def __init__(self, ni, no, kernel_size, strides = (1,1), dilation_rate = (1,1)):
        super(ConvBnPrelu, self).__init__()
        self.conv2d = nn.Conv2d(ni, no, kernel_size = kernel_size, padding = 'same', stride = strides, dilation = dilation_rate)
        self.bn = nn.BatchNorm2d(no,eps = 1e-3, momentum = 0.99, track_running_stats = True, affine = True)
        self.prelu = nn.PReLU(init = 0, num_parameters = no)
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class MainNet(nn.Module): # autoria André
    def __init__(self, m0 = 0.0, var0 = 1.0):
        super(MainNet, self).__init__()
        self.m0 = m0
        self.var0 = var0
        
        # Feature extraction VGG
        self.conv1_1 = ConvBnPrelu(1, 64, (3,3))
        self.conv1_2 = ConvBnPrelu(64, 64, (3,3))
        self.conv2_1 = ConvBnPrelu(64, 128, (3,3))
        self.conv2_2 = ConvBnPrelu(128, 128, (3,3))
        self.conv3_1 = ConvBnPrelu(128, 256, (3,3))
        self.conv3_2 = ConvBnPrelu(256, 256, (3,3))
        self.conv3_3 = ConvBnPrelu(256, 256, (3,3))
        
        # multi-scale ASPP
        
        self.conv4_1 = ConvBnPrelu(256, 256, (3,3), dilation_rate = (1,1)) # scale_1
        self.convori_1_1 = ConvBnPrelu(256, 128, (1,1))
        self.ori_1_2 = nn.Conv2d(128, 90, (1,1), padding = 'same')
        self.convseg_1_1 = ConvBnPrelu(256, 128, (1,1))
        self.seg_1_2 = nn.Conv2d(128, 1, (1,1), padding = 'same')
        
        self.conv4_2 = ConvBnPrelu(256, 256, (3,3), dilation_rate = (4,4)) #scale_2
        self.convori_2_1 = ConvBnPrelu(256, 128, (1,1))
        self.ori_2_2 = nn.Conv2d(128, 90, (1,1), padding = 'same')
        self.convseg_2_1 = ConvBnPrelu(256, 128, (1,1))
        self.seg_2_2 = nn.Conv2d(128, 1, (1,1), padding = 'same')
        
        self.conv4_3 = ConvBnPrelu(256, 256, (3,3), dilation_rate = (8,8)) #scale_3
        self.convori_3_1 = ConvBnPrelu(256, 128, (1,1))
        self.ori_3_2 = nn.Conv2d(128, 90, (1,1), padding = 'same')
        self.convseg_3_1 = ConvBnPrelu(256, 128, (1,1))
        self.seg_3_2 = nn.Conv2d(128, 1, (1,1), padding = 'same')
        
        # ----------------------------------------------------------------------------
        # enhance part
        filters_cos, filters_sin = gabor_bank(stride=2, Lambda=8) # unnecessary when simply deploying with pre-trained weights
        self.enh_img_real_1 = nn.Conv2d(1, 90, (25,25), padding = 'same')
        self.enh_img_imag_1 = nn.Conv2d(1, 90, (25,25), padding = 'same')
        
        self.enh_img_real_1.weight.data = torch.from_numpy(filters_cos).permute(3,2,0,1).double()
        self.enh_img_imag_1.weight.data = torch.from_numpy(filters_sin).permute(3,2,0,1).double()
        
        self.enh_img_real_1.bias.data = torch.zeros(self.enh_img_real_1.bias.data.shape).double()
        self.enh_img_imag_1.bias.data = torch.zeros(self.enh_img_imag_1.bias.data.shape).double()
        
        # ----------------------------------------------------------------------------
        # mnt part
        self.convmnt_1_1 = ConvBnPrelu(2, 64, (9,9))
        self.convmnt_2_1 = ConvBnPrelu(64, 128, (5,5))
        self.convmnt_3_1 = ConvBnPrelu(128, 256, (3,3))
        
        self.convmnt_o_1_1 = ConvBnPrelu(346, 256, (1,1)) # ni = 256 + 90 = n_channels(mnt_conv) + n_channels(ori_out_1)
        self.mnt_o_1_2 = nn.Conv2d(256, 180, (1,1), padding = 'same')
        
        self.convmnt_w_1_1 = ConvBnPrelu(256, 256, (1,1))
        self.mnt_w_1_2 = nn.Conv2d(256, 8, (1,1), padding = 'same')
        
        self.convmnt_h_1_1 = ConvBnPrelu(256, 256, (1,1))
        self.mnt_h_1_2 = nn.Conv2d(256, 8, (1,1), padding = 'same')
        
        self.convmnt_s_1_1 = ConvBnPrelu(256, 256, (1,1))
        self.mnt_s_1_2 = nn.Conv2d(256, 1, (1,1), padding = 'same')
        
        
        
    def forward(self, x):
        bn_image = pytorch_img_normalization(x, self.m0, self.var0)
        # Feature extraction VGG
        conv = self.conv1_1(bn_image)
        conv = self.conv1_2(conv)
        conv = nn.MaxPool2d(kernel_size=2, stride=2)(conv)
        
        conv = self.conv2_1(conv)
        conv = self.conv2_2(conv)
        conv = nn.MaxPool2d(kernel_size=2, stride=2)(conv)
        
        conv = self.conv3_1(conv)
        conv = self.conv3_2(conv)
        conv = self.conv3_3(conv)
        conv = nn.MaxPool2d(kernel_size=2, stride=2)(conv)
        
        # multi-scale ASPP
        
        scale1 = self.conv4_1(conv)
        ori_1 = self.convori_1_1(scale1)
        ori_1 = self.ori_1_2(ori_1)
        seg_1 = self.convseg_1_1(scale1)
        seg_1 = self.seg_1_2(seg_1)
        
        scale2 = self.conv4_2(conv)
        ori_2 = self.convori_2_1(scale2)
        ori_2 = self.ori_2_2(ori_2)
        seg_2 = self.convseg_2_1(scale2)
        seg_2 = self.seg_2_2(seg_2)
        
        scale3 = self.conv4_3(conv)
        ori_3 = self.convori_3_1(scale3)
        ori_3 = self.ori_3_2(ori_3)
        seg_3 = self.convseg_3_1(scale3)
        seg_3 = self.seg_3_2(seg_3)
        
        # sum fusion for ori
    
        ori_out = ori_1 + ori_2 + ori_3
        ori_out1 = nn.Sigmoid()(ori_out)
        ori_out2 = nn.Sigmoid()(ori_out)
        
        # sum fusion for segmentation
         
        seg_out = seg_1 + seg_2 + seg_3
        seg_out = nn.Sigmoid()(seg_out)
        
        # ----------------------------------------------------------------------------
        # enhance part
        
        filter_img_real = self.enh_img_real_1(x)
        filter_img_imag = self.enh_img_imag_1(x)
        
        ori_peak = torch_ori_highest_peak(ori_out1)
        
        ori_peak = torch_select_max(ori_peak)
        upsample_ori = nn.functional.upsample(ori_peak, scale_factor = 8)
        seg_round = nn.Softsign()(seg_out)
        upsample_seg = nn.functional.upsample(seg_round, scale_factor = 8)
        mul_mask_real = filter_img_real * upsample_ori
        enh_img_real = mul_mask_real.sum(axis = 1, keepdims = True)
        mul_mask_imag = filter_img_imag * upsample_ori
        enh_img_imag = mul_mask_imag.sum(axis = 1, keepdims = True)
        enh_img = torch_atan2(enh_img_imag, enh_img_real)
        
        enh_seg_img = torch.cat((enh_img, upsample_seg), 1)
        
        # ----------------------------------------------------------------------------
        # mnt part
        mnt_conv = self.convmnt_1_1(enh_seg_img)
        mnt_conv = nn.MaxPool2d(kernel_size=2, stride=2)(mnt_conv)
        
        mnt_conv = self.convmnt_2_1(mnt_conv)
        mnt_conv = nn.MaxPool2d(kernel_size=2, stride=2)(mnt_conv)
        
        mnt_conv = self.convmnt_3_1(mnt_conv)
        mnt_conv = nn.MaxPool2d(kernel_size=2, stride=2)(mnt_conv)
        
        mnt_o_1 = torch.cat((mnt_conv, ori_out1), 1)
        
        mnt_o_2 = self.convmnt_o_1_1(mnt_o_1)
        mnt_o_3 = self.mnt_o_1_2(mnt_o_2)
        mnt_o_out = nn.Sigmoid()(mnt_o_3)
        
        mnt_w_1 = self.convmnt_w_1_1(mnt_conv)
        mnt_w_2 = self.mnt_w_1_2(mnt_w_1)
        mnt_w_out = nn.Sigmoid()(mnt_w_2)
        
        mnt_h_1 = self.convmnt_h_1_1(mnt_conv)
        mnt_h_2 = self.mnt_h_1_2(mnt_h_1)
        mnt_h_out = nn.Sigmoid()(mnt_h_2)
        
        mnt_s_1 = self.convmnt_s_1_1(mnt_conv)
        mnt_s_2 = self.mnt_s_1_2(mnt_s_1)
        mnt_s_out = nn.Sigmoid()(mnt_s_2)
        
        return ori_out1, ori_out2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out