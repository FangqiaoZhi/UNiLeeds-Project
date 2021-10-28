import itertools
import numpy as np
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as tmf
import pandas as pd


# OUTPUT 388, PADDING 92, INPUT 572
from code.config import OUTPUT_SIZE, PADDING_SIZE, INPUT_SIZE

from code.static import device

from code.trainer import predict_method_with_threshold

def _draw_one_area_helper(data_patch_with_padding, threshold):
    data_patch = data_patch_with_padding[0:3,PADDING_SIZE:OUTPUT_SIZE+PADDING_SIZE,PADDING_SIZE:OUTPUT_SIZE+PADDING_SIZE]
    data_patch = data_patch.permute(1, 2, 0)
    with torch.no_grad():
        # make a batch with only one patch
        data_patch_with_padding_gpu = torch.stack((data_patch_with_padding.to(device),))
        _, predicted = predict_method_with_threshold(data_patch_with_padding_gpu, threshold)
        predict = predicted[0].cpu().float()
    return data_patch, predict


def draw_one_area(dataset_test, index, threshold):
    data_patch_with_padding, label_patch, predict_patch = dataset_test.get_one_area(index)

    data_patch, predict = _draw_one_area_helper(data_patch_with_padding, threshold)
    
    predict = torch.stack((predict, predict, predict),2)
    label_patch = torch.stack((label_patch, label_patch, label_patch),2)
    
    return data_patch, label_patch, predict 


def _merge_to_one_pic_helper(pic_list, m, n):
    return torch.cat((
        torch.cat((pic_list[0], pic_list[1][:,OUTPUT_SIZE-(n-OUTPUT_SIZE):,:]),1),
        torch.cat((pic_list[2][OUTPUT_SIZE-(m-OUTPUT_SIZE):,:,:], pic_list[3][OUTPUT_SIZE-(m-OUTPUT_SIZE):,OUTPUT_SIZE-(n-OUTPUT_SIZE):,:]),1),
    ),0)


'''
def draw_full2(dataset_test, index):
    input_plt_format_list, m, n, orig, labels = dataset_test.get_corners(index)

    pics, imgs = [], []
    for p in input_plt_format_list:
        input_plt_format, predicted = draw_one_area_helper(p)
        img = torch.stack((predicted, predicted, predicted),2)
        pics.append(input_plt_format)
        imgs.append(img)

    return (
        _merge_to_one_pic_helper(pics, m, n),
        torch.stack((labels, labels, labels),2),
        _merge_to_one_pic_helper(imgs, m, n),
    )

'''

def _merge_patches_helper(patches, m, n):
    for i, p in enumerate(patches, 0):
        patches[i] = torch.cat(tuple(patches[i]),1)
    return torch.cat(tuple(patches), 0)


def _merge_patches_to_full(patches, m, n):
    dm = 0
    for i, p in enumerate(patches, 0):
        dn = 0
        for j, q in enumerate(p, 0):
            if dn + OUTPUT_SIZE > n:
                patches[i][j] = patches[i][j][:,OUTPUT_SIZE-(n-dn):,:]
            dn = dn + OUTPUT_SIZE
        patches[i] = torch.cat(tuple(patches[i]),1)
        if dm + OUTPUT_SIZE > m:
            patches[i] = patches[i][OUTPUT_SIZE-(m-dm):,:,:]
        dm = dm + OUTPUT_SIZE
    return torch.cat(tuple(patches), 0)
     
    
def draw_patches(dataset_test, index, max_threshold):
    all_patches_with_padding_in_one_data, all_patches_in_one_label, height, width, one_data, one_label = dataset_test.get_patches(index)
    m = (height-1)//OUTPUT_SIZE + 1
    n = (width-1)//OUTPUT_SIZE + 1
    #print('draw_patches', m, n, height, width, len(all_patches_with_padding_in_one_data), len(all_patches_in_one_label))
    original_patches = [[]for p in range(m)]
    predicted_patches = [[]for p in range(m)]
    for i, p in enumerate(all_patches_with_padding_in_one_data, 0):
        #print('i,p', i, m, n, len(original_patches), len(predicted_patches))
        data_patch, predict = _draw_one_area_helper(p, max_threshold)
        predict = torch.stack((predict, predict, predict),2)
        original_patches[i//n].append(data_patch)
        predicted_patches[i//n].append(predict)

    return (
        _merge_patches_helper(original_patches, height, width),
        torch.stack((one_label, one_label, one_label),2),
        _merge_patches_helper(predicted_patches, height, width),
    )
    
    
def draw_full(dataset_test, index, max_threshold):
    all_patches_with_padding_in_one_data, all_patches_in_one_label, height, width, one_data, one_label = dataset_test.get_patches(index)
    m = (height-1)//OUTPUT_SIZE + 1
    n = (width-1)//OUTPUT_SIZE + 1
    original_patches = [[]for p in range(m)]
    predicted_patches = [[]for p in range(m)]
    for i, p in enumerate(all_patches_with_padding_in_one_data, 0):
        data_patch, predict = _draw_one_area_helper(p, max_threshold)
        predict = torch.stack((predict, predict, predict),2)
        original_patches[i//n].append(data_patch)
        predicted_patches[i//n].append(predict)

    return (
        _merge_patches_to_full(original_patches, height, width),
        torch.stack((one_label, one_label, one_label),2),
        _merge_patches_to_full(predicted_patches, height, width),
    )

