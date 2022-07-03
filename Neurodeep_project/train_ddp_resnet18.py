# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:19:35 2022

@author: adywi
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler

import os, fnmatch
path = os.getcwd()

import numpy as np
from utilities.Dataset import NeuroData
from utilities.Dataset import BrainDataset

from models.model import DeepBrain

import torch
import torch.optim as optim
import torch.nn as nn
from models.resnet import generate_model

import time

from utilities.Savemodel import SaveBestModel
from models.resnet18 import S3ConvXFCResnet
import matplotlib.pyplot as plt

from torch.utils.data import random_split

t0 = time.time()

torch.cuda.empty_cache()

## Load model

model = S3ConvXFCResnet(27,7)


##Destination fichier

file_list = []


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

print(len(file_list))
print(len(label_list))

def load_data(file_list, label_list):
    train_l = file_list[:20465]
    train_lab = label_list[:20465]
    train2_l = file_list[:150]
    train2_lab = label_list[:150]
    val_l = file_list[20465:23350]
    val_lab = label_list[20465:23350]
    test_l = file_list[23350:29197]
    test_lab = label_list[23350:29197]
    train=NeuroData(train_l, train_lab)
    val=NeuroData(val_l, val_lab)
    test=NeuroData(test_l, test_lab)
    return train, val

train_set, val_set = load_data(file_list,label_list)

print(len(train_set))

torch.backends.cudnn.benchmark=True

import ast

parame = open("param10.txt", "r")
parame = parame.read()
parame = ast.literal_eval(parame)


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), # or any optimizer you prefer
                        lr=parame.get("lr", 0.001), # 0.001 is used if no lr is specified
                        momentum=parame.get("momentum", 0.9)
  )

scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parame.get("step_size", 30)),
      gamma=parame.get("gamma", 1.0),  # default is no learning rate decay
  )
#epochs = parame.get("num_epochs", 20) # Play around with epoch number
epochs = 30
# initialize SaveBestModel class

save_best_model = SaveBestModel()


def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    dataset = dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False, sampler=sampler)

    return dataloader


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    
##Save model
def save_model(epochs, model, optimizer, criterion, scheduler):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'model_state_dict': model.state_dict(),
                'loss': criterion,
                }, 'outputs/checkpoint.pth')

def main(rank, world_size):

    dtype=torch.float
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs))

    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    train_loader = prepare(rank, world_size, train_set, batch_size=38)
    val_loader = prepare(rank, world_size, val_set, batch_size=38)

    # instantiate the model(it's your own model) and move it to the right device
    modelddp = model.to(rank)
    modelddp = modelddp.to(dtype)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    modelddp = DDP(modelddp, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    #################### The above is defined previously

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []


    start_time_sec = time.time()

    total_step = len(train_loader)

    for epoch in range(epochs):

        modelddp.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for i, (images, labels) in enumerate(train_loader):
           images = images.cuda(non_blocking=True)
           images = images.to(dtype)
           images = images.to(rank)
           labels = labels.cuda(non_blocking=True)
           labels = labels.type(torch.LongTensor)
           labels = labels.to(rank)

          # Forward pass
           outputs = modelddp(images)
           loss = criterion(outputs, labels)

           # Backward and optimize
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           scheduler.step()

           if rank == 0:
               save_model(epoch, modelddp,optimizer, criterion, scheduler)

           train_loss         += loss.data.item() * images.size(0)
           num_train_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
           num_train_examples += images.shape[0]


        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_loader.dataset)



       # --- EVALUATE ON VALIDATION SET -------------------------------------
        modelddp.eval()

        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for i, (images, labels) in enumerate(val_loader):


           images = images.cuda(non_blocking=True)
           images = images.to(dtype)
           images = images.to(rank)
           labels = labels.cuda(non_blocking=True)
           labels = labels.type(torch.LongTensor)
           labels = labels.to(rank)

          # Forward pass
           outputs = modelddp(images)
           loss = criterion(outputs, labels)

           val_loss         += loss.data.item() * images.size(0)
           num_val_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
           num_val_examples += images.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_loader.dataset)


        
        if rank == 0:
            print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            save_best_model(val_loss, epoch, modelddp, optimizer, criterion, scheduler)
            print(epoch, file=open("outputs/epochs.txt", "a"))
            print(history, file=open("outputs/history.txt", "a"))


            acc = history['acc']
            print(acc, file=open("outputs/acc.txt", "a"))

            val_acc = history['val_acc']
            print(val_acc, file=open("outputs/val_acc.txt", "a"))

            loss = history['loss']
            print(loss, file=open("outputs/loss.txt", "a"))

            val_loss = history['val_loss']
            print(val_loss, file=open("outputs/val_loss.txt", "a"))


        # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    cleanup()
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(main, world_size)