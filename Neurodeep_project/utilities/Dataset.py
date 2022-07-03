 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:51:51 2022

@author: neurodeep
"""

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from models.guided_backprop import GuidedBackprop
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.image import mean_img
from torch import nn
import torch
from nilearn.image import concat_imgs


class BrainDataset(Dataset):
    def __init__(self, file_list, label_list, is_train=True):
        self.is_train = is_train
        self.input_shape = 27
        self.file_list = file_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        img = self.normalize_data(nib.load(self.file_list[index]).get_fdata())
        target = self.label_list[index]
        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def normalize_data(self, data):
        data = data[8:-8, 8:-8, :-10, :self.input_shape]
        data = data / data.max(axis=3)[:, :, :, np.newaxis]
        data[~ np.isfinite(data)] = 0
        return data.transpose(3, 0, 1, 2)
    
class BrainDataset2(Dataset):
    def __init__(self, file_list, label_list, is_train=True):
        self.is_train = is_train
        self.input_shape = 27
        self.file_list = file_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        img = self.normalize_data(nib.load(self.file_list[index]).get_fdata())
        target = self.label_list[index]
        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def normalize_data(self, data):
        data = data / data.max(axis=3)[:, :, :, np.newaxis]
        data[~ np.isfinite(data)] = 0
        return data.transpose(3, 0, 1, 2)
    
class NeuroDataset(Dataset):
    def __init__(self, file_list, label_list, mean, std, is_train=True):
        self.is_train = is_train
        self.input_shape = 27
        self.file_list = file_list
        self.label_list = label_list
        self.mean = mean
        self.std = std
        
    def __getitem__(self, index):
        img = self.normalize_data(nib.load(self.file_list[index]).get_fdata())
        target = self.label_list[index]
        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def normalize_data(self, data):
        data = data[8:-8, 8:-8, :-10, :self.input_shape]
        data = (data-self.mean)/self.std
        return data.transpose(3, 0, 1, 2)
    
class NeuroData(Dataset):
    def __init__(self, file_list, label_list, is_train=True):
        self.is_train = is_train
        self.file_list = file_list
        self.label_list = label_list

        
    def __getitem__(self, index):
        img = self.load(np.load(self.file_list[index]))
        target = self.label_list[index]
        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def load(self, data):
        data = data
        return data.transpose(3, 0, 1, 2)


def visualise(file_list, label_list, model, device):
    template = load_mni152_template()
    feature_list = []
    device = device
    for file, label in zip(file_list, label_list):

        affine = nib.load(file).affine
        file = [file]
        label = [label]
        dataset = BrainDataset2(file, label, is_train=False)
        
        _ = model.eval()


        # Remove LogSoftmax
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        GBP = GuidedBackprop(model)

        inputs, label = dataset[0]

        input_img = nn.Parameter(torch.FloatTensor(inputs).unsqueeze(0), requires_grad=True).to(device)

        guided_grads = GBP.generate_gradients(input_img, label)

        export_gradient = np.zeros((27, 91, 109, 91))
        export_gradient[:, 8:-8, 8:-8, :-10] = guided_grads
        nifti_img = nib.Nifti1Image(export_gradient.transpose(1, 2, 3, 0), affine)
        averaged = mean_img(nifti_img)
        resampled = resample_to_img(averaged, template)
           
        feature_list.append(resampled)
    
    concatened = concat_imgs(feature_list)
    mean = mean_img(concatened)
    
    return mean


def visualise2(file_list, label_list, model, device):
    template = load_mni152_template()
    feature_list = []
    device = device
    for file, label in zip(file_list, label_list):

        affine = nib.load(file).affine
        file = [file]
        label = [label]
        dataset = BrainDataset2(file, label, is_train=False)
        
        _ = model.eval()


        # Remove LogSoftmax
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        GBP = GuidedBackprop(model)

        inputs, label = dataset[0]

        input_img = nn.Parameter(torch.FloatTensor(inputs).unsqueeze(0), requires_grad=True).to(device)

        guided_grads = GBP.generate_gradients(input_img, label)

        export_gradient = np.zeros((27, 91, 109, 91))
        export_gradient[:, 8:-8, 8:-8, :-10] = guided_grads
        nifti_img = nib.Nifti1Image(export_gradient.transpose(1, 2, 3, 0), affine)
        averaged = mean_img(nifti_img)
        resampled = resample_to_img(averaged, template)
           
        feature_list.append(resampled)
    
    concatened = concat_imgs(feature_list)
    mean = mean_img(concatened)

    return mean
