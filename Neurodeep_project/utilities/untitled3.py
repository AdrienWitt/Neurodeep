#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:35:51 2022

@author: neurodeep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:02:54 2022

@author: neurodeep
"""

import os, fnmatch
path = os.getcwd()

import numpy as np
from utilities.Dataset import NeuroData
from models.model import DeepBrain
import torch
import torch.optim as optim
import torch.nn as nn
import ax
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate


from utilities.Savemodel import SaveBestModel

import matplotlib.pyplot as plt

from torch.utils.data import random_split

##Import and prepare data

file_list = []

for path, folders, files in os.walk(path):
    for file in files:
        if fnmatch.fnmatch(file, '*transformed.npy'):
            file_list.append(os.path.join(path, file))

   
label_list = []

for file in file_list:
    if fnmatch.fnmatch(file, '*fear.transformed.npy*' ):
        label = [1,0,0,0,0,0,0]
        label_list.append(label)
    elif fnmatch.fnmatch(file, '*loss.transformed.npy*' ):
        label = [0,1,0,0,0,0,0]
        label_list.append(label)
    elif fnmatch.fnmatch(file, '*present-story.transformed.npy*' ):
            label = [0,0,1,0,0,0,0]
            label_list.append(label)
    elif fnmatch.fnmatch(file, '*rh.transformed.npy*' ):
        label = [0,0,0,1,0,0,0]    
        label_list.append(label)
    elif fnmatch.fnmatch(file, '*relation.transformed.npy*' ):
             label = [0,0,0,0,1,0,0]
             label_list.append(label)
    elif fnmatch.fnmatch(file, '*mental.transformed.npy*' ):
             label = [0,0,0,0,0,1,0]
             label_list.append(label)
    elif fnmatch.fnmatch(file, '*2bk-places.transformed.npy*' ):
                label = [0,0,0,0,0,0,1]
                label_list.append(label)
         
label_list = np.array(label_list)

##Define classes and correct predictions for each classe
classes = ('Emotion', 'Gambling', 'Language', 'Motor', 'Relational', 'Working Memory', 'Social')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
##Classes


model = DeepBrain().to(device)
model.load_state_dict(torch.load('.models/checkpoint_24.pth.tar')['state_dict'])


# validation
def validate(model, testloader, criterion, dtype, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for img, target in test_loader:
            counter += 1
            
            img = img.to(dtype=dtype, device=device)
            target = target.to(device=device)
            # forward pass
            outputs = model(img)
            # calculate the loss
            loss = criterion(outputs, target)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == target).sum().item()
            # collect the correct predictions for each class
            for label, prediction in zip(target, preds):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

epochs = 1
criterion = nn.CrossEntropyLoss()
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valid_loss =  []
valid_acc =  []



for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    valid_epoch_loss, valid_epoch_acc = validate(model, test_loader, criterion, dtype, device)
    valid_loss.append(valid_epoch_loss)
    valid_acc.append(valid_epoch_acc)
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    # save the best model till now if we have the least loss in the current epoch

    print('-'*50)