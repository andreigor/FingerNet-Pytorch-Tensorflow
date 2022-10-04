"""
 * brief:  Dataset class for pre-processing fingernet labels
 *
 * author: André Nóbrega
 * date:   may 25, 2022
"""
import torch
import numpy as np
from torch_snippets import *
from scipy import signal, ndimage

from utils import mnt_reader, get_orientation

def mnt_prep(mnt):
    """
    Transformations required to prepare minutiae labels (x,y) and orientation
    The image is separated in 8x8 patches
    """
    minutiae_w = (torch.zeros((1, int(768/8), int(800/8)))-1).double()
    minutiae_h = (torch.zeros((1, int(768/8), int(800/8)))-1).double()
    minutiae_o = (torch.zeros((1, int(768/8), int(800/8)))-1).double()
    
    
    minutiae_w[0, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int)] = torch.from_numpy(mnt[:, 0] % 8).double()
    minutiae_h[0, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int)] = torch.from_numpy(mnt[:, 1] % 8).double()
    minutiae_o[0, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int)] = torch.from_numpy(mnt[:, 2]).double()

    return minutiae_w, minutiae_h, minutiae_o

def seg_prep(seg):
    """
    Transformations to prepare segmentation label. 
    Basicaly separates the image in patches and applies a global threshold binarization
    """
    label_seg = seg[0, ::8, ::8]
    label_seg[label_seg>0] = 1
    label_seg[label_seg<=0] = 0
    return label_seg

class DatasetImage(Dataset): # there are three mandatory functions: init, len, getitem
    def __init__(self, dataset, transform=None):
        # it gets the image true labels and set the preprocessing transformation
        self.transform = transform
        self.images, self.seg_labels, self.ori_labels, self.mnt_labels = dataset
    def __len__(self): return len(self.images)        
    def __getitem__(self, ix): # returns the item at position ix
        image    = read(self.images[ix])
        seg      = read(self.seg_labels[ix])
        ori      = read(self.ori_labels[ix])
        mnts = np.array(mnt_reader(self.mnt_labels[ix]), dtype = float)
        if self.transform:
            image = self.transform(image).double()
            seg   = self.transform(seg).double()
            ori   = self.transform(ori).double()
        
        minutiae_w, minutiae_h, minutiae_o = mnt_prep(mnts)
        minutiae_w = minutiae_w.unsqueeze(3)
        minutiae_h = minutiae_h.unsqueeze(3)
        minutiae_o = minutiae_o.unsqueeze(3)
        
        
        label_seg = seg_prep(seg).unsqueeze(0)
        minutiae_seg = (minutiae_o!=-1).float()
        
        orientation = get_orientation(ori.unsqueeze(0)) 
        orientation = orientation/np.pi*180+90
        orientation[orientation>=180.0] = 0.0 # orientation [0, 180)
        
        minutiae_o = minutiae_o/np.pi*180+90 # [90, 450)
        minutiae_o[minutiae_o>360] = minutiae_o[minutiae_o>360]-360 # to current coordinate system [0, 360)
        
        minutiae_ori_o = np.copy(minutiae_o) # copy one
        minutiae_ori_o[minutiae_ori_o>=180] = minutiae_ori_o[minutiae_ori_o>=180]-180 # for strong ori label [0,180)  
        
        
        # ori 2 gaussian
        gaussian_pdf = signal.gaussian(361, 3)
        y = np.reshape(np.arange(1, 180, 2), [1,1,1,-1])
        delta = np.array(np.abs(orientation - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori = torch.from_numpy(gaussian_pdf[delta]).squeeze().permute(2,0,1)
        
        # ori_o 2 gaussian
        delta = np.array(np.abs(minutiae_ori_o - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori_o = torch.from_numpy(gaussian_pdf[delta]).squeeze().permute(2,0,1)
        
        
        # mnt_o 2 gaussian
        y = np.reshape(np.arange(1, 360, 2), [1,1,1,-1])
        delta = np.array(np.abs(minutiae_o - y), dtype=int)  
        delta = np.minimum(delta, 360-delta)+180
        label_mnt_o = torch.from_numpy(gaussian_pdf[delta]).squeeze().permute(2,0,1)
        
        # w 2 gaussian
        gaussian_pdf = signal.gaussian(17, 2)
        y = np.reshape(np.arange(0, 8), [1,1,1,-1])
        delta = (minutiae_w-y+8).numpy().astype(int)
        label_mnt_w = torch.from_numpy(gaussian_pdf[delta]).squeeze().permute(2,0,1)
        
        # h 2 gaussian
        delta = (minutiae_h-y+8).numpy().astype(int)
        label_mnt_h = torch.from_numpy(gaussian_pdf[delta]).squeeze().permute(2,0,1)
        
        # mnt cls label -1:neg, 0:no care, 1:pos
        label_mnt_s = np.copy(minutiae_seg)
        label_mnt_s[label_mnt_s==0] = -1 # neg to -1
        label_mnt_s = (label_mnt_s+ndimage.maximum_filter(label_mnt_s, size=(1,3,3,1)))/2 # around 3*3 pos -> 0
        label_mnt_s = torch.from_numpy(label_mnt_s).squeeze(0).permute(2,0,1)

        # apply segmentation
        
        have_alignment = (torch.sum(label_ori) != 0).double()
        minutiae_seg = minutiae_seg.squeeze(3)
        
        label_ori = label_ori * label_seg * have_alignment
        label_ori_o = label_ori_o * minutiae_seg
        label_mnt_o = label_mnt_o * minutiae_seg
        label_mnt_w = label_mnt_w * minutiae_seg
        label_mnt_h = label_mnt_h * minutiae_seg
        
        return image, label_ori, label_ori_o, label_seg, label_mnt_w, label_mnt_h, label_mnt_o, label_mnt_s


        

def GetBatches(dataset, batchsize, transform = None):
    datatensor = DatasetImage(dataset, transform) 
    dataloader = DataLoader(datatensor, batch_size=batchsize)
    return(dataloader)

prep = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((768,800)),
    transforms.ToTensor()
])