import torch
import numpy as np
from torch import nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ori2angle_ptorch(ori):
    kernal2angle = torch.from_numpy(np.reshape(np.arange(1, 180, 2, dtype=float), [1,90,1,1])/90.*np.pi) #2angle = angle*2
    sin2angle, cos2angle = torch.sin(kernal2angle).to(device), torch.cos(kernal2angle).to(device)
    
    sin2angle_ori = torch.sum(ori*sin2angle, axis = 1, keepdims=True)
    cos2angle_ori = torch.sum(ori*cos2angle, axis = 1, keepdims=True)
    modulus_ori = torch.sqrt(torch.square(sin2angle_ori) + torch.square(cos2angle_ori))
    
    return sin2angle_ori, cos2angle_ori, modulus_ori

class CoherenceLoss(torch.nn.Module):
    """
    Coherence loss function
    """

    def __init__(self):
        super(CoherenceLoss, self).__init__()
    def forward(self, y_true, y_pred):
        # clip
        y_pred = torch.where(y_pred < 1e-7, 1e-7, torch.where(y_pred > (1 - 1e-7), 1 - 1e-7, y_pred))
        # get ROI
        label_seg = torch.sum(y_true, axis = 1, keepdims=True)
        label_seg = (label_seg > 0).double()
        
        #coherence loss
        mean_kernal = torch.from_numpy(np.reshape(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)/8, [1,1,3,3])).double().to(device)
        sin2angle_ori, cos2angle_ori, modulus_ori = ori2angle_ptorch(y_pred)
        
        sin2angle = nn.functional.conv2d(sin2angle_ori, mean_kernal, padding = 'same')
        cos2angle = nn.functional.conv2d(cos2angle_ori, mean_kernal, padding = 'same')
        modulus = nn.functional.conv2d(modulus_ori, mean_kernal, padding = 'same')
        
        coherence = torch.sqrt(torch.square(sin2angle) + torch.square(cos2angle))/ (modulus + 1e-7)
        coherenceloss = torch.sum(label_seg) / (torch.sum(coherence*label_seg) + 1e-7) - 1
        
        return coherenceloss

class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    Weighted cross entropy loss function
    """

    def __init__(self, lambda_pos = 1.0, lambda_neg = 1.0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
    def forward(self, y_true, y_pred):
        # clip
        y_pred = torch.where(y_pred < 1e-7, 1e-7, torch.where(y_pred > (1 - 1e-7), 1 - 1e-7, y_pred))
        # get ROI
        label_seg = torch.sum(y_true, axis = 1, keepdims=True)
        label_seg = (label_seg > 0).double()
        # weighted cross entropy loss
        logloss = self.lambda_pos*y_true*torch.log(y_pred) + self.lambda_neg*(1 - y_true)*torch.log(1 - y_pred)
        logloss = logloss * label_seg
        logloss = -torch.sum(logloss) / (torch.sum(label_seg) + 1e-7)
        
        return logloss

class SegmentationLoss(torch.nn.Module):
    """
    Segmentation loss function.
    Represents the weak segmentation loss, which is equal to a cross entropy loss + smooth loss
    """

    def __init__(self, lamb = 1.):
        super(SegmentationLoss, self).__init__()
        self.lamb = lamb
    def forward(self, y_true, y_pred):
        y_pred = torch.where(y_pred < 1e-7, 1e-7, torch.where(y_pred > (1 - 1e-7), 1 - 1e-7, y_pred))
        label_pos = (y_true > 0).double()
        
        # modified weighted cross entropy part
        total_elements = torch.sum(torch.ones(y_true.shape))
        
        # positive and negative label weights
        lamb_pos = 0.5 * total_elements / torch.sum(label_pos)
        lamb_neg = 1 / (2 - 1/lamb_pos)

        # cross-entropy loss
        logloss = lamb_pos*y_true*torch.log(y_pred) + lamb_neg*(1-y_true)*torch.log(1-y_pred)
        logloss = -torch.mean(torch.sum(logloss, axis = 1))

        # smooth loss
        smooth_kernal = torch.from_numpy(np.reshape(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)/8, [1, 1, 3, 3])).double().to(device)
        smoothloss = torch.mean(torch.abs(nn.functional.conv2d(y_pred, smooth_kernal)))

        loss = logloss + self.lamb*smoothloss

        return loss
    
class MinutiaeScoreLoss(torch.nn.Module):
    """
    Minutiae score loss function.
    Represents the ground truth minutiae label score
    """

    def __init__(self):
        super(MinutiaeScoreLoss, self).__init__()
    def forward(self, y_true, y_pred):
        y_pred = torch.where(y_pred < 1e-7, 1e-7, torch.where(y_pred > (1 - 1e-7), 1 - 1e-7, y_pred))
        label_seg = (y_true != 0).double()
        y_true = torch.where(y_true < 0, 0.0, y_true.double())

        # weighted cross entropy loss
        total_elements = torch.sum(label_seg) + 1e-7
        lamb_pos, lamb_neg = 10., .5
        
        logloss = lamb_pos*y_true*torch.log(y_pred) + lamb_neg*(1-y_true)*torch.log(1-y_pred)

        # apply ROI
        logloss = logloss*label_seg
        logloss = -torch.sum(logloss) / total_elements
        return logloss
