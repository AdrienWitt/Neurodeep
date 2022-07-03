# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:14:17 2022

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
import ax
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate
import time

from utilities.Savemodel import SaveBestModel

import matplotlib.pyplot as plt

from torch.utils.data import random_split

t0 = time.time()

print(torch.cuda.is_available())
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Load files

file_list = []

for path, folders, files in os.walk(path):
    for file in files:
        if fnmatch.fnmatch(file, '*train.npy'):
            file_list.append(os.path.join(path, file))

print(len(file_list))
file_list.sort()

label_list = []

for file in file_list:
    if fnmatch.fnmatch(file, '*gene_train.npy*' ):
            label = 0
            label_list.append(label)
    elif fnmatch.fnmatch(file, '*rap_train.npy*' ):
                label = 1
                label_list.append(label)


## Select samples and smoothed samples so they are in the same set

swar = file_list[0:200]
war = file_list[200:400]

swar_lab = label_list[0:200]
war_lab = label_list[200:400]

train_l = swar[0:140] + war[0:140]
train_lab = swar_lab[0:140] + war_lab[0:140]

print(len(train_l))


val_l = swar[140:160] + war[140:160]
val_lab = swar_lab[140:160] + war_lab[140:160]

## Create sets

train_set = NeuroData(train_l, train_lab)
val_set = NeuroData(val_l, val_lab)

print(len(train_set))
print(len(val_set))

##Define model
model = DeepBrain()
model.load_state_dict(torch.load('./models/checkpoint_24.pth.tar')['state_dict'])


##Freeze param
for param in model.parameters():
    param.requires_grad = False

##Recreate FC layers
model.classifier = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 2),
    nn.LogSoftmax(dim=1))

#Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

##train function
def net_train(net, train_loader, parameters, dtype, device):
  net.to(dtype=dtype, device=device)

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), # or any optimizer you prefer
                        lr=parameters.get("lr", 0.001), # 0.001 is used if no lr is specified
                        momentum=parameters.get("momentum", 0.9)
  )

  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parameters.get("step_size", 30)),
      gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
  )

  num_epochs = parameters.get("num_epochs", 3) # Play around with epoch number
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
                                batch_size=parameterization.get("batchsize", 3),
                                shuffle=True,
                                num_workers=0)

    test_loader = torch.utils.data.DataLoader(val_set,
                                batch_size=parameterization.get("batchsize", 3),
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.04], "log_scale": True},
        {"name": "batchsize", "type": "range", "bounds": [10, 64]},
        {"name": "momentum", "type": "range", "bounds": [0.1, 0.99]},
        {"name": "num_epochs", "type": "range", "bounds": [10, 80]},
        {"name": "step_size", "type": "range", "bounds": [20, 40]},
    ],

    total_trials=20,
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)



#Plot accuracy

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])

best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Classification Accuracy, %",
)
render(best_objective_plot)

#find best hyper parameter

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)

parame=best_arm._parameters
print(param)
print(param, file=open("param.txt", "a"))




