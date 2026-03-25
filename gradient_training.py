#!/usr/bin/env python3
# -*- coding: utf-8 -*-





import random
import os

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


from utils import SimpleCNN, SimpleCNN_deep, SimpleCNN_wide

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




MNIST_train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))


FashionMNIST_train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.286,), (0.353,))
]))




#%% Create validation set for 10 classes from the unused train data

filenames = ['MNIST30.pt', 'MNIST300.pt', 'MNIST600.pt', 
            'FashionMNIST30.pt', 'FashionMNIST300.pt', 'FashionMNIST600.pt']

#filenames= ['MNIST1200.pt'] 

for d in range(len(filenames)):
    
    filename = './datasets/'+ filenames[d]
    tt = torch.load(filename, weights_only = True)
    test_data = tt[0]
    test_target = tt[1]
    train_data = tt[2]
    train_target = tt[3]

    if filenames[d][:3] == 'MNI':
        train_dataset = MNIST_train_dataset
    else:
        train_dataset = FashionMNIST_train_dataset    
    
    Ntrain = train_data.shape[0]

    k= 0 
    validation_data = torch.zeros([len(train_dataset) -Ntrain,1,28,28])
    validation_target = torch.zeros([len(train_dataset) -Ntrain])
    for i in range(len(train_dataset)):
        print(d,i)
        datapoint = train_dataset[i][0]
        is_train_point = False
        for j in range(Ntrain):
            if (datapoint == train_data[j][0]).all():
                is_train_point = True
                break
        if not is_train_point:
            validation_data[k] = datapoint
            validation_target[k] = train_dataset[i][1]
            k += 1
    assert k == len(train_dataset) -Ntrain
            
    val_filename = './datasets/Val_'+ filenames[d] 
    torch.save([validation_data,validation_target], val_filename)



#%% Create validation set for 2 classes from the unused train data

MNIST_train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))


FashionMNIST_train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.286,), (0.353,))
]))


filenames = ['MNIST16.pt', 'FashionMNIST16.pt']


# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = i 
#     img = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     #plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


for d in range(2):
    
    filename = './datasets/'+ filenames[d]
    tt = torch.load(filename, weights_only = True)
    test_data = tt[0]
    test_target = tt[1]
    train_data = tt[2]
    train_target = tt[3]

    if filenames[d][:3] == 'MNI':
        train_dataset = MNIST_train_dataset
    else:
        train_dataset = FashionMNIST_train_dataset    
    
    Ntrain = train_data.shape[0]

    k= 0 
    validation_data = torch.zeros([len(train_dataset) -Ntrain,1,28,28])
    validation_target = torch.zeros([len(train_dataset) -Ntrain])
    for i in range(len(train_dataset)):
        if not train_dataset[i][1] in {0,7}:
            continue
        print(d,i)
        datapoint = train_dataset[i][0]
        is_train_point = False
        for j in range(Ntrain):
            if (datapoint == train_data[j][0]).all():
                is_train_point = True
                break
        if not is_train_point:
            validation_data[k] = datapoint
            validation_target[k] = 0 if train_dataset[i][1] == 0 else 1
            k += 1
    # assert k == len(train_dataset) -Ntrain
            
    val_filename = './datasets/Val_'+ filenames[d] 
    torch.save([validation_data,validation_target], val_filename)

#%%

def set_seed(seed_value=42):
    """Set seeds for reproducibility across all components."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # for multi-GPU setups
        # Optional: set cuDNN to be deterministic for some operations, might impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    



#%% Train with GD

def get_accuracy(model, data, target):
    with torch.no_grad():        
        output = model(data)
    _,pred = output.topk(1)        
    acc = (pred.squeeze() == target).sum().item()                    
    return 100*acc/data.shape[0]



models = [SimpleCNN, SimpleCNN_deep, SimpleCNN_wide]


train_iterations = 5

test_accuracies_GD = np.zeros([8,3,train_iterations]) #[datasets, models, train_iterations ]
test_accuracies_SGD = np.zeros([8,3,train_iterations]) #[datasets, models, train_iterations ]

test_accuracies_SGD2 = np.zeros([8,3,train_iterations]) #[datasets, models, train_iterations ]


filenames = ['MNIST16.pt','FashionMNIST16.pt', 
            'MNIST30.pt', 'MNIST300.pt', 'MNIST600.pt', 
            'FashionMNIST30.pt', 'FashionMNIST300.pt', 'FashionMNIST600.pt']




device = torch.device("cuda:0")    
max_tolerance = 50
batch_sizes = [5, 5, 5, 30, 60, 5, 30, 60]  #for each data set, for SGD
SGD = True


SGD = False
#filenames= ['MNIST2400.pt'] 
#models = [SimpleCNN]

test_accs = np.zeros([1,1,train_iterations])



for d in range(len(filenames)):  # iterate over dataset

    filename = filenames[d]
    
    data_filename = './datasets/'+ filename
    val_filename = './datasets/Val_'+ filename
    
    tt = torch.load(data_filename, weights_only = True)
    test_data = tt[0].to(device)
    test_target = tt[1].to(device)
    train_data = tt[2].to(device)
    train_target = tt[3].to(device)
    
    tv = torch.load(val_filename, weights_only = True)
    val_data = tv[0].to(device)
    val_target = tv[1].to(device)

    for m, mod in enumerate(models):
                
        for tit in range(train_iterations):
        
            tolerance = 0 
            set_seed(d*100 + tit)        
            model = mod()
            model.to(device)
    
            criterion = nn.CrossEntropyLoss()
            
            bin_op = binaryconnect.BC(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    
            best_val_acc_so_far = 0
            test_acc = 0 
            
            print('\n', filename, 'model:', m, 'train_iteration:', tit) 
            for it in range(250000):
                
                bin_op.binarization()        
                
                if it % 50 ==0:
                    e = get_accuracy(model, train_data, train_target)            
                    q = get_accuracy(model, test_data, test_target)    
                    v = get_accuracy(model, val_data, val_target)    
                    
                    if e == 100.0 and v >= best_val_acc_so_far:
                        test_acc = q
                        if v> best_val_acc_so_far:
                            best_val_acc_so_far = v
                            tolerance = 0
                        
                    if it % 500 ==0:
                        print('it', it, 'e:', e, 'v:', v, 'q:', q)
                        if v <= best_val_acc_so_far:
                            tolerance +=1
                            if tolerance == max_tolerance:
                                break
                    
            
                
                if SGD: # sample mini_batch
                    sample_ids = torch.randint(len(train_data), size = (batch_sizes[d],) )                     
                    train_data_batch = train_data[sample_ids]
                    train_target_batch = train_target[sample_ids]                    
                    output = model(train_data_batch)           
                    loss = criterion(output, train_target_batch)
                else:
                    output = model(train_data)           
                    loss = criterion(output, train_target)
                # compute gradient and do update 
                optimizer.zero_grad()
                loss.backward()
                bin_op.restore()
                optimizer.step()
            
            if SGD:
                test_accuracies_SGD2[d,m,tit] = test_acc
            else:
                #test_accuracies_GD[d,m,tit] = test_acc 
                test_accs[d,m,tit] = test_acc 
            print(filename, 'model:', m, 'test_acc:', test_acc) 





# GD was compted with  torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
mus = test_accuracies_GD.mean(2)
stds = test_accuracies_GD.std(2)
lr = 0.0001


mus2 = test_accuracies_SGD.mean(2)
stds = test_accuracies_SGD.std(2)



# Adam, lr = 0.0001
mus3 = test_accuracies_SGD2.mean(2)
std3 = test_accuracies_SGD.std(2)



# eliminate those runs in which test_acc was zero because zero error was not reached
mus_sgd = np.zeros([8,3])
std_sgd = np.zeros([8,3])

for i in range(8):
    for j in range(3):
        mus_sgd[i,j] = (test_accuracies_SGD2[i,j,:][test_accuracies_SGD2[i,j,:]>0]).mean()
        std_sgd[i,j] = (test_accuracies_SGD2[i,j,:][test_accuracies_SGD2[i,j,:]>0]).std()
        



with np.printoptions(precision=2, suppress=True):
    print(mus)
    print(stds)


lr = 0.0001
test_acc_filename = './datasets/test_acc.npz'
np.savez(test_acc_filename, test_accuracies_GD = test_accuracies_GD, test_accuracies_SGD2 = test_accuracies_SGD2)

#res = np.load(test_acc_filename, allow_pickle=True)        



