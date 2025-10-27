#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 19:29:54 2025

@author: aripakman
"""
import torchvision

import binaryconnect

import numpy as np

import torch
import torch.nn as nn

import torch.nn.parallel
import matplotlib.pyplot as plt

import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets



from collections import Counter



# 

from utils import SimpleCNN
from config import n_train, n_test, bins_e, bins_q, eq_limits


####################



#%% Datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load full MNIST
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter only digits 0 and 7
chosen_classes = [0, 7]
train_indices = [i for i, t in enumerate(full_train.targets) if t in chosen_classes]
test_indices = [i for i, t in enumerate(full_test.targets) if t in chosen_classes]

train_subset = torch.utils.data.Subset(full_train, train_indices)
test_subset = torch.utils.data.Subset(full_test, test_indices)

# --- Build small balanced train set: 8 samples of each class (total 16) ---
targets = torch.tensor([full_train.targets[i].item() for i in train_indices])
class0_indices = torch.nonzero(targets == 0).squeeze()
class7_indices = torch.nonzero(targets == 7).squeeze()
n_per_class = 8

sel_indices = torch.cat([
    class0_indices[torch.randperm(len(class0_indices))[:n_per_class]],
    class7_indices[torch.randperm(len(class7_indices))[:n_per_class]]
])

train_subset = torch.utils.data.Subset(train_subset, sel_indices)

# --- Convert to tensors ---
Ntrain = len(train_subset)
Ntest = len(test_subset)

train_data = torch.zeros([Ntrain, 1, 28, 28])
train_target = torch.zeros([Ntrain], dtype=torch.int64)
for i in range(Ntrain):
    img, label = train_subset[i]
    train_data[i] = img
    train_target[i] = 0 if label == 0 else 1   # relabel 0→0, 7→1

test_data = torch.zeros([Ntest, 1, 28, 28])
test_target = torch.zeros([Ntest], dtype=torch.int64)
for i in range(Ntest):
    img, label = test_subset[i]
    test_data[i] = img
    test_target[i] = 0 if label == 0 else 1   # relabel 0→0, 7→1

print(f"Train set: {Ntrain} samples | Test set: {Ntest} samples")
print(f"Unique train labels: {train_target.unique().tolist()}")

# Save tensors for REWL walkers
torch.save([test_data.cpu(), test_target.cpu(), train_data.cpu(), train_target.cpu()],
           './wl_data_tensors.pt')

torch_t_filename = './wl_data_tensors.pt'
tt = torch.load(torch_t_filename)
test_data = tt[0]
test_target = tt[1]
train_data = tt[2]
train_target = tt[3]




# train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=60, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=60, shuffle=True)




#%% Initial weights
    

device = torch.device("cuda:0")    

model = SimpleCNN()
model.to(device)
bin_op = binaryconnect.BC(model)

next(model.parameters())
bin_op.binarization()
next(model.parameters())
bin_op.restore()
next(model.parameters())


test_data = test_data.to(device)
test_target = test_target.to(device)
train_data = train_data.to(device)
train_target =train_target.to(device)


def get_train_accuracy():
    with torch.no_grad():
        output = model(train_data)
    _,pred = output.topk(1)        
    acc = (pred.squeeze() == train_target).sum().item()            
    #e = int((acc/self.n_train)*(self.bins_e-1))        
    e = acc # maximum granularity 
    return e

        
        
def get_test_accuracy():
    with torch.no_grad():
        output = model(test_data)
    _,pred = output.topk(1)        
    acc = (pred.squeeze() == test_target).sum().item()    
    q = int(((acc-1)/n_test)*bins_q)
    return q


e = get_train_accuracy()
q = get_test_accuracy()


criterion = nn.CrossEntropyLoss()


#%%
# We train using BinaryConnect https://arxiv.org/abs/1511.00363


model = SimpleCNN()
model.to(device)
bin_op = binaryconnect.BC(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


for it in range(15000):
    bin_op.binarization()
    if it%100 ==0:
        e = get_train_accuracy()
        q = get_test_accuracy()

    print(e,q)        

    output = model(train_data)           
    loss = criterion(output, train_target)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    bin_op.restore()
    optimizer.step()


#%%


#saving the weights of the model for each rank
model = SimpleCNN()
model.to(device)
bin_op = binaryconnect.BC(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=5e-4)

initialized = set()

for it in range(50000):
    
    bin_op.binarization()
    if it%100 ==0:
        e = get_train_accuracy()
        q = get_test_accuracy()
        print(e,q)        
        
        for rank in range(len(eq_limits)):
            if not rank in initialized:
                if eq_limits[rank]['q_min'] <= q  <= eq_limits[rank]['q_max'] and eq_limits[rank]['e_min'] <= e  <= eq_limits[rank]['e_max']:
                    initialized.add(rank)
                    v_filename = f'./initial_weights/initial_v_{rank}.pt'                    
                    with torch.no_grad():
                        v = torch.nn.utils.parameters_to_vector(model.parameters()).cpu()        
                    torch.save(v.cpu(), v_filename)
                    print('it', it, 'e:', e, 'q:', q, f'Saved v for rank {rank}!')
            else:                   
                print('it', it, 'e:', e, 'q:', q)

    output = model(train_data)
    loss = criterion(output, train_target)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    bin_op.restore()
    optimizer.step()

        
    
    








        

