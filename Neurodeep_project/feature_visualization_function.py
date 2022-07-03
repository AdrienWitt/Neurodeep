# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 22:15:44 2022

@author: adywi
"""


from torch import nn

import torch

import nibabel as nib
from models.guided_backprop import GuidedBackprop
from nilearn.datasets import load_mni152_template
from nilearn.image import mean_img
from torch import nn
from nilearn.image import concat_imgs
from utilities.Dataset import BrainDataset2
import nibabel as nib
from nilearn import plotting
from nilearn import datasets
from nilearn import surface

#file_list = your file list
#label_list = #your label_list
         
#model = yout model


#model.load_state_dict(torch.load(your weights)

def visualise(file_list, label_list, model, device):
    template = load_mni152_template()
    feature_list = []
    device = device
    for file, label in zip(file_list, label_list):
        fmri = nib.load(file)
        affine = fmri.affine
        maxi = fmri.get_fdata().max()
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
        print(maxi)
        export_gradient = guided_grads*maxi
        print(export_gradient.max())

        nifti_img = nib.Nifti1Image(export_gradient.transpose(1, 2, 3, 0), affine)
        averaged = mean_img(nifti_img)
           
        feature_list.append(averaged)
    
    concatened = concat_imgs(feature_list)
    mean = mean_img(concatened)

    return mean



#img = visualise(gene_list, label_list, model, device)

#img.to_filename('img.nii.gz')


#plotting.plot_img_on_surf(img, threshold= your treshold, inflate='True', views=['lateral', 'medial'],   hemispheres=['left', 'right'], colorbar=True, title='Control', surf_mesh='fsaverage')
                       