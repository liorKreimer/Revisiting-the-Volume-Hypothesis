#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
from utils import SimpleCNN_wide
from config import n_train, n_test, bins_e, bins_q, eq_limits


####################



#%% Datasets

# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=transforms.ToTensor()
# )

# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=transforms.ToTensor()
# )



# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()



###########################




train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))



val_dataset = datasets.MNIST('./data', train=False, download=True,transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))



figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(str(label))
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

Counter(train_dataset.targets.numpy())
Counter(val_dataset.targets.numpy())

# Create balanced train dataset
# Number of training samples to use
Ntrain = 2400  #
# Calculate samples per class
samples_per_class = Ntrain // 10  # 3000 // 10 = 300 samples per class
train_data = torch.zeros([Ntrain, 1, 28, 28])
train_target = torch.zeros([Ntrain], dtype=torch.int64)
# Create balanced train dataset
for i in range(10):  # Iterate through 10 classes
    inds = np.asanyarray(train_dataset.targets.numpy() == i).nonzero()[0]
    # Select 'samples_per_class' number of random indices for the current class
    shuffled_inds = inds[torch.randperm(len(inds))[:samples_per_class]]
    for j in range(samples_per_class):  # Loop 'samples_per_class' times
        # Populate train_data and train_target
        train_data[i * samples_per_class + j] = train_dataset[shuffled_inds[j]][0]
        train_target[i * samples_per_class + j] = train_dataset[shuffled_inds[j]][1]


Ntest = len(val_dataset)
test_data = torch.zeros([Ntest,1,28,28])
test_target = torch.zeros([Ntest], dtype =torch.int64)
for i in range(Ntest):
    test_data[i,-1] = val_dataset[i][0]
    test_target[i] = val_dataset[i][1]


# save train-test dataset that will be loaded by each random walker
torch_t_filename = './wl_data_tensors.pt'
torch.save([test_data.cpu(),test_target.cpu(), train_data.cpu(), train_target.cpu()], torch_t_filename)

# load the train-test dataset that will be loaded by each random walker
torch_t_filename = './wl_data_tensors.pt'
tt = torch.load(torch_t_filename)
test_data = tt[0]
test_target = tt[1]
train_data = tt[2]
train_target = tt[3]

# train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=60, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=60, shuffle=True)


# %% Initial weights


device = torch.device("cuda:0")

model = SimpleCNN_wide()
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
train_target = train_target.to(device)


def get_train_accuracy():
    with torch.no_grad():
        output = model(train_data)
    _, pred = output.topk(1)
    acc = (pred.squeeze() == train_target).sum().item()
    # e = int((acc/self.n_train)*(self.bins_e-1))
    e = acc  # maximum granularity
    return e


def get_test_accuracy():
    with torch.no_grad():
        output = model(test_data)
    _, pred = output.topk(1)
    acc = (pred.squeeze() == test_target).sum().item()
    q = int(((acc - 1) / n_test) * bins_q)
    return q


e = get_train_accuracy()
q = get_test_accuracy()

criterion = nn.CrossEntropyLoss()

# %%
# We train using BinaryConnect https://arxiv.org/abs/1511.00363


model = SimpleCNN_wide()
model.to(device)
bin_op = binaryconnect.BC(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

for it in range(1):
    bin_op.binarization()
    if it % 100 == 0:
        e = get_train_accuracy()
        q = get_test_accuracy()

    print(e, q)

    output = model(train_data)
    loss = criterion(output, train_target)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    bin_op.restore()
    optimizer.step()

# %%


# saving the weights of the model for each rank
model = SimpleCNN_wide()
model.to(device)
bin_op = binaryconnect.BC(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)

initialized = set()

for it in range(50000):

    bin_op.binarization()
    if it % 100 == 0:
        e = get_train_accuracy()
        q = get_test_accuracy()
        print(e, q)

        for rank in range(len(eq_limits)):
            if not rank in initialized:
                if eq_limits[rank]['q_min'] <= q <= eq_limits[rank]['q_max'] and eq_limits[rank]['e_min'] <= e <= \
                        eq_limits[rank]['e_max']:
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














