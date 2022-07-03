# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:24:49 2022

@author: adywi
"""


import os, fnmatch
path = os.getcwd()

import numpy as np
from utilities.Dataset import NeuroData
from models.model import DeepBrain
import torch
import torch.optim as optim
import torch.nn as nn

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate
import time
from utilities.Savemodel import SaveBestModel
from models.resnet18 import S3ConvXFCResnet

import matplotlib.pyplot as plt

from torch.utils.data import random_split

file_list = []

print(torch.cuda.is_available())


model = S3ConvXFCResnet(27,7)

if torch.cuda.device_count() > 1:
     print("Let's use", torch.cuda.device_count(), "GPUs!", file=open("multigpu.txt", "a"))
     print("Let's use", torch.cuda.device_count(), "GPUs!")
     #dim = -1 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
     model = nn.DataParallel(model)
else :
    print("1GPU")
    print("1gpu", file=open("1gpu.txt", "a"))
    
    

## Load files

for path, folders, files in os.walk(path):
    for file in files:
        if fnmatch.fnmatch(file, '*transformed_max.npy'):
            file_list.append(os.path.join(path, file))

file_list.sort()
label_list = []

for file in file_list:
    if fnmatch.fnmatch(file, '*fear.transformed_max.npy*' ):
        label = 0
        label_list.append(label)
    elif fnmatch.fnmatch(file, '*loss.transformed_max.npy*' ):
        label = 1
        label_list.append(label)
    elif fnmatch.fnmatch(file, '*present-story.transformed_max.npy*' ):
            label = 2
            label_list.append(label)
    elif fnmatch.fnmatch(file, '*rh.transformed_max.npy*' ):
        label = 3
        label_list.append(label)
    elif fnmatch.fnmatch(file, '*relation.transformed_max.npy*' ):
             label = 4
             label_list.append(label)
    elif fnmatch.fnmatch(file, '*mental.transformed_max.npy*' ):
             label = 5
             label_list.append(label)
    elif fnmatch.fnmatch(file, '*2bk-places.transformed_max.npy*' ):
            label = 6
            label_list.append(label)

label_list = np.array(label_list)
print(len(label_list), len(file_list))

def load_data(file_list, label_list):
    train_l = file_list[:20465]
    train_lab = label_list[:20465]
    train2_l = file_list[:3957]
    train2_lab = label_list[:3957]
    val_l = file_list[20465:21027]
    val_lab = label_list[20465:21027]
    test_l = file_list[21027:29197]
    test_lab = label_list[21027:29197]
    train=NeuroData(train2_l, train2_lab)
    val=NeuroData(val_l, val_lab)
    test=NeuroData(test_l, test_lab)
    return(train, val)

train_set, val_set = load_data(file_list,label_list)

##Train function for optimization

##train function
def net_train(net, train_loader, parameters, dtype, device):
  net.to(dtype=dtype, device=device)

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), # or any optimizer you prefer
                        lr=parameters.get("lr", 0.001), # 0.001 is used if no lr is specified
                        momentum=parameters.get("momentum", 0.99)
  )

  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parameters.get("step_size", 30)),
      gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
  )

  num_epochs = parameters.get("num_epochs", 50) # Play around with epoch number
  # Train Network
  for _ in range(num_epochs):
      for inputs, labels in train_loader:
          # move data to proper dtype and device
          inputs = inputs.to(dtype=dtype, device=device)
          labels = labels.type(torch.LongTensor)
          labels = labels.to(device=device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()
  return net

def train_evaluate(parameterization):

    # constructing a new training data loader allows us to tune the batch size
    train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=parameterization.get("batchsize", 32),
                                shuffle=True,
                                num_workers=0)

    test_loader = torch.utils.data.DataLoader(val_set,
                                batch_size=parameterization.get("batchsize", 32),
                                shuffle=True,
                                num_workers=0)

    # Get neural net
    untrained_net = model

    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader,
                            parameters=parameterization, dtype=dtype, device=device)

    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
    )#, trained_net, train_loader, test_loader


#torch.cuda.set_device(0)
dtype=torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

t0 = time.time()

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.04], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.1, 0.99]},
        {"name": "batchsize", "type": "range", "bounds": [10, 80]},
        {"name": "num_epochs", "type": "range", "bounds": [10, 50]},
        {"name": "step_size", "type": "range", "bounds": [20, 40]},
    ],

    total_trials=10,
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)


#find best hyper parameter

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)

param=best_arm._parameters
print(param)
print(param, file=open("param.txt", "a"))