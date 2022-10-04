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

from utils import *
from modelTrain import MainNet
from datasetTrain import DatasetImage, GetBatches, prep
from criterion import WeightedCrossEntropyLoss, SegmentationLoss, MinutiaeScoreLoss, CoherenceLoss

coherence_loss = CoherenceLoss()
weighted_cross_entropy_loss = WeightedCrossEntropyLoss()
segmentation_loss = SegmentationLoss()
minutiae_score_loss = MinutiaeScoreLoss()


def train_batch(model, data, optimizer):

    image, label_ori, label_ori_o, label_seg, label_mnt_w, label_mnt_h, label_mnt_o, label_mnt_s = [t.to(device) for t in data]
    optimizer.zero_grad()
    ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = model(image)

    
    weak_orientation_loss   = .1 * (coherence_loss(label_ori, ori_out_1) + weighted_cross_entropy_loss(label_ori, ori_out_1))
    strong_orientation_loss = .1 * (weighted_cross_entropy_loss(label_ori_o, ori_out_2))
    weak_segmentation_loss  = 10 * (segmentation_loss(label_seg, seg_out))
    ground_truth_mnt_o_loss = .5 * (weighted_cross_entropy_loss(label_mnt_o, mnt_o_out))
    ground_truth_mnt_w_loss = .5 * (weighted_cross_entropy_loss(label_mnt_w, mnt_w_out))
    ground_truth_mnt_h_loss = .5 * (weighted_cross_entropy_loss(label_mnt_h, mnt_h_out))
    ground_truth_mnt_s_loss = .5 * (minutiae_score_loss(label_mnt_s, mnt_s_out))

    total_loss = weak_orientation_loss + strong_orientation_loss + weak_segmentation_loss + ground_truth_mnt_h_loss + ground_truth_mnt_o_loss + ground_truth_mnt_s_loss + ground_truth_mnt_w_loss
    total_loss.backward()

    return total_loss.item()

def valid_batch(model, data):
    image, label_ori, label_ori_o, label_seg, label_mnt_w, label_mnt_h, label_mnt_o, label_mnt_s = [t.to(device) for t in data]
    model.eval()
    ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = model(image)

    weak_orientation_loss   = .1 * (coherence_loss(label_ori, ori_out_1) + weighted_cross_entropy_loss(label_ori, ori_out_1))
    strong_orientation_loss = .1 * (weighted_cross_entropy_loss(label_ori_o, ori_out_2))
    weak_segmentation_loss  = 10 * (segmentation_loss(label_seg, seg_out))
    ground_truth_mnt_o_loss = .5 * (weighted_cross_entropy_loss(label_mnt_o, mnt_o_out))
    ground_truth_mnt_w_loss = .5 * (weighted_cross_entropy_loss(label_mnt_w, mnt_w_out))
    ground_truth_mnt_h_loss = .5 * (weighted_cross_entropy_loss(label_mnt_h, mnt_h_out))
    ground_truth_mnt_s_loss = .5 * (minutiae_score_loss(label_mnt_s, mnt_s_out))

    total_loss = weak_orientation_loss + strong_orientation_loss + weak_segmentation_loss + ground_truth_mnt_h_loss + ground_truth_mnt_o_loss + ground_truth_mnt_s_loss + ground_truth_mnt_w_loss
    model.train()

    return total_loss.item()

def main():
    if len(sys.argv) != 3:
        print("Usage: FingernetTrain <...>\n"
	       "[1] input folder with training dataset. It must contain the following sub-folders \n"
           "\t-images (latent training images)\n"
           "\t-ori_labels (aligned references)\n"
           "\t-seg labels (latent segmentation mask)\n"
           "\t-mnt_labels (manually marked txts files)\n"
	       "[2] output model name\n",
	       "main");
        exit()
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using', device, 'device')

    # Loading Model
    model_path =     '../models/fingernet.pth'
    model = MainNet().double().to(device)
    print('Importando pesos pré-treinados Fingernet...\n')
    model.load_state_dict(torch.load(model_path))

    # Loading dataset
    dataset    = sys.argv[1]
    train_perc = 0.9
    images     = Glob(dataset + '/images/' + '*')
    seg_labels = Glob(dataset + '/seg_labels/' +  '*')
    ori_labels = Glob(dataset + '/ori_labels/' + '*')
    mnt_labels = Glob(dataset + '/mnt_labels/' + '*.mnt')
    num_train_samples = int(len(images)*train_perc)
    train_images, valid_images           = images[:num_train_samples], images[num_train_samples:]
    train_seg_labels, valid_seg_labels   = seg_labels[:num_train_samples], seg_labels[num_train_samples:]
    train_ori_labels, valid_ori_labels   = ori_labels[:num_train_samples], ori_labels[num_train_samples:]
    train_mnt_labels, valid_mnt_labels   = mnt_labels[:num_train_samples], mnt_labels[num_train_samples:]

    train_fileset    = (train_images, train_seg_labels, train_ori_labels, train_mnt_labels)
    valid_fileset    = (valid_images, valid_seg_labels, valid_ori_labels, valid_mnt_labels)

    
    batch_size = 1
    trainload = GetBatches(train_fileset, batch_size, prep)
    validload = GetBatches(valid_fileset, batch_size, prep)
    print('Quantidade de batches de treino:', len(trainload))
    print('Quantidade de batches de validacao:', len(validload))
    nepochs = 100
    optimizer  = optim.Adam(model.parameters(),lr = 0.00001, weight_decay=0.01)
    for i, (name, param) in enumerate(model.named_parameters()):
        param.requires_grad = False
        if (i == 34): # only freezing the first conv layers
            break


    print('Starting model training ...')
    tic = time.time()

    # Training loop
    log     = Report(nepochs)
    for epoch in range(nepochs):
        N = len(trainload)
        for i, data in enumerate(trainload):
            batch_loss = train_batch(model, data, optimizer)
            log.record(epoch+(1+i)/N, trn_loss=batch_loss, end='\r')
        N = len(validload)
        with torch.no_grad():
            for i,data in enumerate(validload):
                batch_loss = valid_batch(model, data)
                log.record(epoch+(1+i)/N, val_loss = batch_loss, end ='\r')
    # saving model
    torch.save(model.state_dict(), f'../models/{sys.argv[2]}.pth')

    # saving training loss
    with open(f'training_loss_history_{sys.argv[2]}.txt', 'w') as file:
        for trn_loss in log.history('trn_loss'):
            file.write(str(trn_loss) + '\n')

    with open(f'valid_loss_history_{sys.argv[2]}.txt', 'w') as file:
        for val_loss in log.history('val_loss'):
            file.write(str(val_loss) + '\n')
   
    toc = time.time()
    print('Training finished with sucess in {:.2f}s'.format(toc - tic))




if __name__ == '__main__':
    main()
