# -*- coding: utf-8 -*-

import os, fnmatch
path = os.getcwd()

import numpy as np
from utilities.Dataset import NeuroData
from models.model import DeepBrain
import torch
import torch.optim as optim
import torch.nn as nn

import time
from utilities.Savemodel import SaveBestModel

import matplotlib.pyplot as plt

from torch.utils.data import random_split

t0 = time.time()

torch.cuda.empty_cache()


dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_list = []


## Load files

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


t2 = time.time()

## Load hyperparameters

import ast
parame = open("param.txt", "r")
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
epochs = 500 # Play around with epoch number

# initialize SaveBestModel class

save_best_model = SaveBestModel()

# start the training
train_loader = torch.utils.data.DataLoader(train_set,
                             batch_size=parame.get("batchsize", 3),
                             shuffle=True,
                             num_workers=0)

test_loader = torch.utils.data.DataLoader(val_set,
                             batch_size=parame.get("batchsize", 3),
                             shuffle=True,
                             num_workers=0)



def train(model, optimizer, scheduler, loss_fn, train_dl, val_dl, epochs, dtype, device):

    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()

    model.to(dtype=dtype, device=device)

    for epoch in range(1, epochs+1):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
                                                                                                  
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(dtype=dtype, device=device)
            y    = batch[1].type(torch.LongTensor)
            y    = batch[1].to(device=device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        model.to(device)
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0


        for batch in val_dl:

            x    = batch[0].to(dtype=dtype, device=device)
            y    = batch[1].type(torch.LongTensor)
            y    = batch[1].to(device=device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)


        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        save_best_model(val_loss, epoch, model, optimizer, criterion, scheduler)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history


history = train(
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
    loss_fn = criterion,
    train_dl = train_loader,
    val_dl = test_loader, dtype=dtype,
    epochs = epochs,
    device=device)       



## Plots

import matplotlib.pyplot as plt

acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('outputs/accuracy.png')
plt.close()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('outputs/loss.png')




##Save model
def save_model(epochs, model, optimizer, criterion, scheduler):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')


save_model(epochs, model, optimizer, criterion, scheduler)

print(history, file=open("outputs/history.txt", "a"))
