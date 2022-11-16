"""
 * brief:  Applies a pre-trained FingerNet model to a given folder with fingerprint images. Images must be in grayscale.
           For further reference check https://github.com/592692070/FingerNet
 *
 * author: André Nóbrega
 * date:   may 17, 2022
 * return: output folder with segmented images
"""

import os, sys
import time

import torch
from   torch_snippets import *
from torchvision    import transforms
import cv2
import imageio

from utils import *
from modelDeploy import MainNet
import settings



class DatasetImage(Dataset): # there are three mandatory functions: init, len, getitem
    def __init__(self, dataset, transform=None):
        # it gets the image true labels and set the preprocessing transformation
        self.dataset   = dataset
        self.transform = transform
    def __len__(self): return len(self.dataset)        
    def __getitem__(self, ix): # returns the item at position ix
        
        filename = self.dataset[ix]
        image    = read(filename)
        if self.transform:
            image = self.transform(image).double()
            
        return(image)

def GetBatches(dataset, batchsize, transform = None):
    datatensor = DatasetImage(dataset, transform) 
    dataloader = DataLoader(datatensor, batch_size=batchsize)
    return(dataloader)

def get_image_prep(image_shape):
    # it is important that the image dimensions are a multiple of 8
    
    # getting closest 8 multiple for x dimension
    x_pad = 8 - image_shape[0] % 8
    y_pad = 8 - image_shape[1] % 8
    # width, height = int(image_shape[0] / 8) * 8, int(image_shape[1] / 8) * 8
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad((y_pad, x_pad, 0, 0), fill = 255),
        # transforms.Resize((width,height)),
        transforms.ToTensor()
    ])

    return prep


def main():
    if len(sys.argv) != 3:
        print("Usage: FingerNetSegmentation <...>\n"
	       "[1] input folder with fingerprint(s) image(s) \n"
	       "[2] output folder with the object saliency map\n",
	       "main");
        exit()
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)
    settings.globals_init()

    start_time = time.time()

    # Loading Model
    model_path =     '../models/fingernet.pth'
    model = MainNet().double().to(device)
    model.load_state_dict(torch.load(model_path))
    # Loading dataset
    folder     = sys.argv[1]
    file_ext   = '*'
    fileset    = Glob(folder + '/' + file_ext)
    batch_size = 1
    

    ix = 0

    # applying model forward and saving to output folder

    try:
        os.mkdir(sys.argv[2])
    except FileExistsError:
        print(f'Warning: "{sys.argv[2]}" folder already exists. May overwrite files.\n')
    
    for ix,path in enumerate(fileset):
        files = [path]

        dataset = DatasetImage(files, None)
        image_shape = dataset[0].shape
        prep = get_image_prep(image_shape)
        dataloader = GetBatches(files, batch_size, prep)
        print(f'Applying model to image {ix}/{int(len(fileset)/batch_size)} ...')
        with torch.no_grad():
            for batch in dataloader:
                            
                tic = time.time()
                
                enh_img_real, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = model(batch.to(device))
                
                
                # ####################### POST PROCESSING ###########################
                tic = time.time()

                # transforming torch tensors to numply to reuse fingernet original util functions
                seg_out = seg_out.permute(0,2,3,1).detach().cpu().numpy()
                mnt_o_out = mnt_o_out.permute(0,2,3,1).detach().cpu().numpy()
                mnt_s_out = mnt_s_out.permute(0,2,3,1).detach().cpu().numpy()
                mnt_w_out = mnt_w_out.permute(0,2,3,1).detach().cpu().numpy()
                mnt_h_out = mnt_h_out.permute(0,2,3,1).detach().cpu().numpy()
                enh_img_real = enh_img_real.permute(0,2,3,1).detach().cpu().numpy()

                # post processing segmentation mask
                round_seg = np.round(np.squeeze(seg_out))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
                seg_out = cv2.morphologyEx(round_seg, cv2.MORPH_OPEN, kernel)

                # post processing minutia labels
                mnt = label2mnt(np.squeeze(mnt_s_out)*np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5)
                mnt_nms = nms(mnt)

                # post processing orientation field
                ori = torch_ori_highest_peak(ori_out_1)
                ori = ((torch.argmax(ori, axis = 1)*2-90)/180.*np.pi).detach().cpu().numpy()
                
                # images saving parameters
                img_name = str(fileset[ix]).split('/')[-1].split('.')[0]
                set_name = '' # str(fileset[ix]).split('/')[2]
                # img_size = np.array(batch.shape[2:], dtype=np.int32)/8*8 be careful here
                img_size = image_shape
                image = batch.permute(0,2,3,1).detach().cpu().numpy()
                output_dir = sys.argv[2]

                # Saving images
                mnt_writer(mnt_nms, img_name, img_size, "%s/%s/%s.mnt"%(output_dir, set_name, img_name))
                draw_ori_on_img(image, ori, seg_out, "%s/%s/%s_ori.png"%(output_dir, set_name, img_name))        
                draw_minutiae(image, mnt_nms[:,:3], "%s/%s/%s_mnt.png"%(output_dir, set_name, img_name))
                imageio.imsave("%s/%s/%s_enh.png"%(output_dir, set_name, img_name), (np.squeeze(enh_img_real)*ndimage.zoom(np.round(np.squeeze(seg_out)), [8,8], order=0)))
                imageio.imsave("%s/%s/%s_seg.png"%(output_dir, set_name, img_name), (ndimage.zoom(np.round(np.squeeze(seg_out)), [8,8], order=0))) 
                print(f'Post processing finised with sucess in {time.time() - tic} for image {ix + 1}/{int(len(fileset)/batch_size)}')
                print('####################################')



    print('Deploy finished with sucess in {:.2f}s'.format(time.time() - start_time))




if __name__ == '__main__':
    main()
