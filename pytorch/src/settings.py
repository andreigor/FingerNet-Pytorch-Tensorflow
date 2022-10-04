import torch
import numpy as np
from scipy import signal
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gausslabel(length=180, stride=2):

    gaussian_pdf = signal.gaussian(length+1, 3)
    label = np.reshape(np.arange(stride/2, length, stride), [1,1,-1,1])
    y = np.reshape(np.arange(stride/2, length, stride), [1,1,1,-1])
    delta = np.array(np.abs(label - y), dtype=int)
    delta = np.minimum(delta, length-delta)+length/2
    return gaussian_pdf[delta.astype(int)]


def gaussian2d(shape=(5,5),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def globals_init():
    global glabel
    global zero_tensor
    global one_tensor
    global gaussian

    glabel = torch.tensor(gausslabel(length=180,stride=2), device = torch.device(device)).permute(3,2,0,1).double()
    zero_tensor = torch.tensor([0], device = torch.device(device)).double()
    one_tensor = torch.tensor([0.999], device = torch.device(device)).double()
    gaussian = torch.from_numpy(np.reshape(gaussian2d((5, 5), 1), [1,1,5,5])).to(device)

