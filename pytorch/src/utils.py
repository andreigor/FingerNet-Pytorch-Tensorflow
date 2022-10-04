import math
import numpy as np
import torch
from   torch import nn as nn
from scipy import signal, sparse, spatial, ndimage
import matplotlib.pyplot as plt
import time

import settings



device = 'cuda' if torch.cuda.is_available() else 'cpu'
def pytorch_img_normalization(im_input, m0 = 0.0, var0 = 1.0): # autoria Enrico
    
    im_input = torch.clone(im_input)
    m = torch.mean(im_input)
    var = torch.var(im_input)

    im_output =  torch.where(im_input > m, 
                (m0 + (torch.sqrt((var0*(im_input-m)*(im_input-m))/var))), 
                (m0-(torch.sqrt((var0*(im_input-m)*(im_input-m))/var))))
                
    return im_output

def torch_atan2(enh_img_imag, enh_img_real):
    y,x = (enh_img_imag), (enh_img_real + 1e-7)
    atan = torch.atan(y/x)
    angle = torch.where(x > settings.zero_tensor, atan, settings.zero_tensor)
    cond1 = x < settings.zero_tensor
    cond2 = y >= settings.zero_tensor
    cond3 = y < settings.zero_tensor
    angle = torch.where(cond1&cond2, atan + np.pi, angle)
    angle = torch.where(cond1&cond3, atan - np.pi, angle)
    return angle


def torch_select_max(x):
    x = x / (x.max(axis = 1, keepdims = True).values + 1e-7)
    x = torch.where(x > settings.one_tensor, x, settings.zero_tensor)
    x = x / (x.sum(axis = 1, keepdims = True) + 1e-7)
    return x

def torch_ori_highest_peak(y_pred, length=180):
    ori_gau = nn.functional.conv2d(y_pred, settings.glabel,padding='same')
    return ori_gau


# currently can only produce one each time
def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5):
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    assert len(mnt_s_out.shape)==2 and len(mnt_w_out.shape)==3 and len(mnt_h_out.shape)==3 and len(mnt_o_out.shape)==3 
    # get cls results
    mnt_sparse = sparse.coo_matrix(mnt_s_out>thresh)
    mnt_list = np.array(list(zip(mnt_sparse.row, mnt_sparse.col)), dtype=np.int32)
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 4))
    # get regression results
    mnt_w_out = np.argmax(mnt_w_out, axis=-1)
    mnt_h_out = np.argmax(mnt_h_out, axis=-1)
    mnt_o_out = np.argmax(mnt_o_out, axis=-1) # TODO: use ori_highest_peak(np version)
    # get final mnt
    mnt_final = np.zeros((len(mnt_list), 4))
    mnt_final[:, 0] = mnt_sparse.col*8 + mnt_w_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 1] = mnt_sparse.row*8 + mnt_h_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 2] = (mnt_o_out[mnt_list[:,0], mnt_list[:,1]]*2-89.)/180*np.pi
    mnt_final[mnt_final[:, 2]<0.0, 2] = mnt_final[mnt_final[:, 2]<0.0, 2]+2*np.pi
    mnt_final[:, 3] = mnt_s_out[mnt_list[:,0], mnt_list[:, 1]]
    return mnt_final

def angle_delta(A, B, max_D=np.pi*2):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D-delta)
    return delta

def distance(y_true, y_pred, max_D=16, max_O=np.pi/6):
    D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], 'euclidean')
    O = spatial.distance.cdist(np.reshape(y_true[:, 2], [-1, 1]), np.reshape(y_pred[:, 2], [-1, 1]), angle_delta)
    return (D<=max_D)*(O<=max_O)

def nms(mnt):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=np.pi/6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(np.bool), :]

def mnt_writer(mnt, image_name, image_size, file_name):
    f = open(file_name, 'w')
    f.write('%s\n'%(image_name))
    f.write('%d %d %d\n'%(mnt.shape[0], image_size[0], image_size[1]))
    for i in range(mnt.shape[0]):
        f.write('%d %d %.6f\n'%(mnt[i,0], mnt[i,1], mnt[i,2]))
    f.close()
    return

def mnt_reader(file_name):
    f = open(file_name)
    minutiae = []
    for i, line in enumerate(f):
        if i < 2 or len(line) == 0: continue
        w, h, o, s = [float(x) for x in line.split()]
        w, h = int(round(w)), int(round(h))
        minutiae.append([w, h, o])
    f.close()
    return minutiae

def draw_ori_on_img(img, ori, mask, fname, coh=None, stride=16):
    ori = np.squeeze(ori)
    mask = np.squeeze(np.round(mask))
    img = np.squeeze(img)
    ori = ndimage.zoom(ori, np.array(img.shape)/np.array(ori.shape, dtype=float), order=0)
    if mask.shape != img.shape:
        mask = ndimage.zoom(mask, np.array(img.shape)/np.array(mask.shape, dtype=float), order=0)
    if coh is None:
        coh = np.ones_like(img)
    fig = plt.figure()
    plt.imshow(img,cmap='gray')
    # plt.hold(True)  
    for i in range(stride,img.shape[0],stride):
        for j in range(stride,img.shape[1],stride):
            if mask[i, j] == 0:
                continue
            x, y, o, r = j, i, ori[i,j], coh[i,j]*(stride*0.9)
            plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
    plt.axis([0,img.shape[1],img.shape[0],0])
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)            
    return

def draw_minutiae(image, minutiae, fname, r=15):
    image = np.squeeze(image)
    fig = plt.figure()
    plt.imshow(image,cmap='gray')
    # plt.show()
    plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
    for x, y, o in minutiae:
        plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
    plt.axis([0,image.shape[1],image.shape[0],0])
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    return


def GetPadConfig(input_size, kernel, stride):
    output_size = int(math.ceil(float(input_size) / float(stride)))
    pad_total = int((output_size - 1) * stride + kernel - input_size)
    pad_left = int(pad_total / 2)
    pad_right = pad_total - pad_left
    return pad_left, pad_right

# traditional orientation estimation -> gradient method
def get_orientation(image, stride = 8, window = 17):
    strides = (stride, stride)
    E = torch.ones([1,1,window,window]).double()
    sobelx = torch.from_numpy(np.reshape(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float), [1,1,3,3]))
    sobely = torch.from_numpy(np.reshape(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float), [1,1,3,3]))

    Ix = nn.functional.conv2d(image, sobelx, stride = 1, padding = 'same')
    Iy = nn.functional.conv2d(image, sobely, stride = 1, padding = 'same')

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # padding Ix2, Iy2 and Ixy
    P_top,P_bot = GetPadConfig(Ix2.shape[2], window, stride)
    P_left,P_right = GetPadConfig(Ix2.shape[3], window, stride)
    Ix2 = nn.functional.pad(Ix2,(P_left,P_right,P_top,P_bot))
    
    P_top,P_bot = GetPadConfig(Iy2.shape[2], window, stride)
    P_left,P_right = GetPadConfig(Iy2.shape[3], window, stride)
    Iy2 = nn.functional.pad(Iy2,(P_left,P_right,P_top,P_bot))
    
    P_top,P_bot = GetPadConfig(Ixy.shape[2], window, stride)
    P_left,P_right = GetPadConfig(Ixy.shape[3], window, stride)
    Ixy = nn.functional.pad(Ixy,(P_left,P_right,P_top,P_bot))

    
    Gxx = nn.functional.conv2d(Ix2, E, stride = strides)
    Gyy = nn.functional.conv2d(Iy2, E, stride = strides)
    Gxy = nn.functional.conv2d(Ixy, E, stride = strides)

    Gxx_yy = Gxx - Gyy
    theta = torch_atan2(2*Gxy, Gxx_yy) + torch.pi

    phi_x = nn.functional.conv2d(torch.cos(theta), settings.gaussian, padding = 'same')
    phi_y = nn.functional.conv2d(torch.sin(theta), settings.gaussian, padding = 'same')

    theta = torch_atan2(phi_y, phi_x)/2
    
    return theta.permute(0,2,3,1).cpu().numpy()

def gabor_fn(ksize, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    # Bounding box
    nstds = 3
    xmax = ksize[0]/2
    ymax = ksize[1]/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb_cos = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    gb_sin = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / Lambda * x_theta + psi)
    return gb_cos, gb_sin
    
def gabor_bank(stride=2,Lambda=8):
    filters_cos = np.ones([25,25,int(180/stride)], dtype=float)
    filters_sin = np.ones([25,25,int(180/stride)], dtype=float)
    for n, i in enumerate(range(-90,90,stride)):  
        theta = i*np.pi/180.
        kernel_cos, kernel_sin = gabor_fn((24,24),4.5, -theta, Lambda, 0, 0.5)
        filters_cos[..., n] = kernel_cos
        filters_sin[..., n] = kernel_sin
    filters_cos = np.reshape(filters_cos,[25,25,1,-1])
    filters_sin = np.reshape(filters_sin,[25,25,1,-1])
    return filters_cos, filters_sin