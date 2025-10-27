#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist





# model class 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1, bias = False)  # Convolutional layer
        #self.bn1 = nn.BatchNorm2d(6, affine = False)  # Batch normalization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling
        self.fc1 = nn.Linear(6 * 14 * 14, 64, bias = False)  # Fully connected layer
        self.bn2 = nn.BatchNorm1d(64, affine=False)  # Batch normalization            
        self.fc2 = nn.Linear(64, 2, bias= False)  # Output layer

    def forward(self, x):
        #x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        return x



class Walker:
    def __init__(self, limits, params, device, rank, model_class = SimpleCNN):
        
        self.device = device
        self.model_class = model_class
        self.rank = rank        
        # load dataset and move to device
        torch_t_filename = './wl_data_tensors.pt'
        self.results_folder = os.environ["WLRESULTS"]  # Get the results folder path from sbatch env variable
        tt = torch.load(torch_t_filename)
        self.test_data = tt[0].to(device)
        self.test_target = tt[1].to(device)
        self.train_data = tt[2].to(device)
        self.train_target = tt[3].to(device)
        

        self.e_min = limits['e_min']
        self.e_max = limits['e_max']
        self.q_min = limits['q_min']
        self.q_max = limits['q_max']
        
        self.n_train = params.n_train
        self.n_test = params.n_test
        self.bins_e = params.bins_e
        self.bins_q = params.bins_q
        
        
        self.prop_size = 3 # number of spins to flip in each proposal
        
        
        
        self.model = model_class()
        self.model.to(device)

        # load initial weights 
        v_filename = f'./initial_weights/initial_v_{rank}.pt'
        if os.path.exists(v_filename):            
            self.v = torch.load(v_filename)        
            self.v = self.v.to(device)
            self.initialize_parameters()
            self.need_initialization =  torch.zeros(1, device='cpu') #recall bool(0) == False
            
        else:
            with torch.no_grad():
                self.v = torch.nn.utils.parameters_to_vector(self.model.parameters())                    
            self.need_initialization = torch.ones(1, device='cpu') #recall bool(1) == True

        samples_filename = f'./results/rank_{rank}_results.npz'
        #samples_filename = f'{self.results_folder}/rank_{self.rank}_results.npz'
        if self.need_initialization or not os.path.exists(samples_filename):     
            # define container for current histogram
            self.h = np.zeros([self.e_max-self.e_min+1, self.q_max-self.q_min+1 ])            
            # define container for log_density of states 
            self.log_g = np.zeros([self.e_max-self.e_min+1, self.q_max-self.q_min+1 ])            
            self.log_f = 1
            self.update_its = {}
            self.it = 0
        else:
            res = np.load(samples_filename, allow_pickle=True)
            self.h = res['h']
            self.log_g = res['log_g']
            self.log_f = res['log_f'].item()
            self.it = res['it'].item()
            self.update_its = res['update_its'].item()


        self.num_spins = len(self.v)
        self.random_same = 0
        self.reject_out  = 0
        self.random_accept = 0


    def initialize_parameters(self):
        
        with torch.no_grad():        
            torch.nn.utils.vector_to_parameters(self.v, self.model.parameters())                

        e = self.get_train_accuracy()
        q = self.get_test_accuracy()      
        
        # check that the initial weights were correct for this walker
        assert self.e_min <= e <= self.e_max and self.q_min <= q <= self.q_max                               
        
        self.ie = e - self.e_min
        self.iq = q - self.q_min            


    def save(self):

        if not self.need_initialization:
            samples_filename = f'{self.results_folder}/rank_{self.rank}_results.npz'
            np.savez(samples_filename, h=self.h, log_g = self.log_g, it=self.it, log_f = self.log_f, update_its=self.update_its )


        # loaded_arrays = np.load(samples_filename, allow_pickle=True)
        # h = loaded_arrays['h']
        # log_g = loaded_arrays['log_g']

        

    def get_train_accuracy(self):
        with torch.no_grad():
            output = self.model(self.train_data)
        _,pred = output.topk(1)        
        acc = (pred.squeeze() == self.train_target).sum().item()            
        #e = int((acc/self.n_train)*(self.bins_e-1))        
        e = acc # maximum granularity 
        return e
        
        
    def get_test_accuracy(self):
        with torch.no_grad():
            output = self.model(self.test_data)
        _,pred = output.topk(1)        
        acc = (pred.squeeze() == self.test_target).sum().item()
        q = int(((acc-1)/self.n_test)*self.bins_q)        
        return q
        
        
    def reset(self):
        torch.nn.utils.vector_to_parameters(self.v, self.model.parameters())
        self.ie = self.get_train_accuracy() - self.e_min
        self.iq = self.get_test_accuracy() - self.q_min
        
        if self.ie < 0 or self.ie > self.e_max-self.e_min or self.iq < 0 or self.iq > self.q_max-self.q_min:            
            self.v[self.js] *= -1  # undo last spin flip 

        
    def step(self):
    
        if self.need_initialization: 
            return
        
        self.it +=1
        self.js = np.random.choice(self.num_spins, size=self.prop_size, replace=False)            
        self.v[self.js] *= -1
    
        torch.nn.utils.vector_to_parameters(self.v, self.model.parameters())        
        ie_new = self.get_train_accuracy() - self.e_min
        iq_new = self.get_test_accuracy() - self.q_min    
    
        if self.ie == ie_new and self.iq == iq_new:
            self.random_same+=1

        
        if ie_new < 0 or ie_new > self.e_max-self.e_min:
            reject = True
            self.reject_out +=1
            
        elif iq_new < 0 or iq_new > self.q_max-self.q_min:
            reject = True
            self.reject_out +=1

        else:
            lg = self.log_g[self.ie,self.iq]
            lg_new = self.log_g[ie_new,iq_new]        
            log_MH = lg - lg_new   
            
            if log_MH >= 0  or np.random.rand() < np.exp(log_MH):
                # accept proposal        
                self.ie = ie_new 
                self.iq = iq_new                 
                self.random_accept +=1
                reject = False 
            else:
                reject = True
            
        if reject:
            self.v[self.js] *= -1  # undo the spin flip 
            # note that when the proposal is rejected, self.v does not agree with model.parameters(), but that is not a problem
    
        self.log_g[self.ie,self.iq] += self.log_f    
        self.log_g -= self.log_g.max()            
        self.h[self.ie,self.iq] +=1 
        
        
        if (self.h > 0.90*self.h.mean()).all():            
            self.h = 0*self.h             
            self.update_its[self.log_f] = self.it
            self.log_f = 0.5*self.log_f 
            print('f: ', self.log_f)


def exchange(walker,paired_rank):
    
    rank = walker.rank
    
    if paired_rank <0:  # in some cases, there are ranks (in the grid boundary) which do not try to exchange
        return                        
    
    pair_need_initialization =  torch.zeros(1, device='cpu')
    
    if rank > paired_rank:
        dist.send(walker.need_initialization, dst = paired_rank)
        dist.recv(pair_need_initialization, src=paired_rank)                       
    else:
        dist.recv(pair_need_initialization, src=paired_rank)                       
        dist.send(walker.need_initialization, dst = paired_rank)
        
    if not walker.need_initialization and pair_need_initialization:
        print('Sending my weights to my paired rank to initialize')
        
        v = walker.v.to('cpu')
        dist.send(v, dst = paired_rank)
        
        return 
    
    if walker.need_initialization and  not pair_need_initialization:  
        
        v = torch.zeros(walker.num_spins, device = 'cpu')
        dist.recv(v, src=paired_rank)               
        walker.v = v.to(walker.device)
               
        
        with torch.no_grad():        
            torch.nn.utils.vector_to_parameters(walker.v, walker.model.parameters())                

        e = walker.get_train_accuracy()
        q = walker.get_test_accuracy()      
        
        print('I received the weights from the paired rank to initialize', f'e: {e}, q:{q}')
        
        if walker.e_min <= e <= walker.e_max and walker.q_min <= q <= walker.q_max:                               
            walker.initialize_parameters()            
            v_filename = f'./initial_weights/initial_v_{rank}.pt'
            torch.save(walker.v.cpu(), v_filename)

            # === NEW LINE TO ADD ===
            # Copy the file from the scratch space to your home folder's initial_weights directory
            os.system(f'cp {v_filename} {os.environ["HOME"]}/rewl_Project/initial_weights/')
            # === END OF NEW LINE ===

            print(f'Walker {rank} initialized from walker {paired_rank}!')            
            walker.need_initialization =  torch.zeros(1, device='cpu')  #recall bool(1) == True
        else:
            print('The paired rank was out of my range')
        return 
        
    if walker.need_initialization and  pair_need_initialization:  
        return 
    
    # If the code reaches this point, both walkers are initialized
     
    # Get absolute indices to send to the paired rank 
    e = walker.ie + walker.e_min                
    q = walker.iq + walker.q_min                                
    e = torch.tensor(e, device='cpu', dtype = torch.int64)
    q = torch.tensor(q, device='cpu', dtype = torch.int64)
                    
    # prepare containers to receive (e,q) from the paired rank
    pe = torch.zeros(1, device='cpu', dtype = torch.int64)
    pq = torch.zeros(1, device='cpu', dtype = torch.int64)
    
        
    if rank > paired_rank:
        dist.send(e, dst = paired_rank)
        dist.send(q, dst = paired_rank)
        dist.recv(pe, src=paired_rank)                    
        dist.recv(pq, src=paired_rank)                    
    else:
        dist.recv(pe, src=paired_rank)                    
        dist.recv(pq, src=paired_rank)                    
        dist.send(e, dst = paired_rank)
        dist.send(q, dst = paired_rank)

    
    pe = int(pe)
    pq = int(pq)
    
    # the MH acceptance decision is performed by the higher rank
    if rank > paired_rank:
        
        # the lower ranks informs
        out_of_range = torch.zeros(1, device="cpu")
        dist.recv(out_of_range, src=paired_rank)        
        
        if out_of_range: 
            print('I am out of the range of my paired walker', f'e: {e}, q: {q}')
            return     # I am out of the range of my pair            
        else:   

            MH_reject = False                                    # maybe exchange
            pair_log_prob = torch.zeros(1, device="cpu", dtype = torch.float64)
            dist.recv(pair_log_prob, src=paired_rank)                                            
            pair_log_prob = pair_log_prob.item()                                
            
            
            
            # check if paired walker is outside my window
            if pe < walker.e_min or pe > walker.e_max or pq < walker.q_min or pq > walker.q_max:
                MH_reject = True
                print('My paired walker is out of my range', f'pe: {pe}, pq: {pq}')
            else:
                
                my_log_prob = walker.log_g[walker.ie,walker.iq] -walker.log_g[pe-walker.e_min, pq - walker.q_min]
                
                print('received pair log prob', pair_log_prob, 'my log prob', my_log_prob )                
                
                logMH = my_log_prob + pair_log_prob
                        
                if logMH >= 0 or np.random.rand() < np.exp(logMH):   # exchange 
                    # inform to the paired rank  that there will be exchange                     
                    is_there_exchange = torch.ones(1, device="cpu")                    
                    dist.send(is_there_exchange, dst = paired_rank)                                                
                    
                    print(f'Walker {rank} exchanged with walker {paired_rank}.')
                    
                    # let's interchange!               
                    v = walker.v.cpu()
                    dist.send(v, dst = paired_rank)                            
                    dist.recv(v, src = paired_rank)      
                    walker.v = v.to(walker.device)                    
                    walker.ie = pe -walker.e_min
                    walker.iq = pq -walker.q_min
                    
                else:
                    MH_reject = True
                    print('MH rejected')


            if MH_reject:  # report to the paired rank that there will be no exchange
                is_there_exchange = torch.zeros(1, device="cpu")                
                dist.send(is_there_exchange, dst = paired_rank)                                            

        
    else:   # meanwhile in the lower rank
        if pe < walker.e_min or pe > walker.e_max or pq < walker.q_min or pq > walker.q_max:
            out_of_range = torch.ones(1, device="cpu")
            dist.send(out_of_range, dst = paired_rank)     # no exchange, goodbye.
            
            print(f'The paired rank {paired_rank} is out of my range. pe: {pe}, pq: {pq}') 
            return
            
        else:                                              # maybe exchange
            out_of_range = torch.zeros(1, device="cpu")
            dist.send(out_of_range, dst = paired_rank)
        
            my_log_prob = walker.log_g[walker.ie,walker.iq] -walker.log_g[pe-walker.e_min, pq - walker.q_min]                        
            my_log_prob = torch.tensor(my_log_prob, device="cpu", dtype = torch.float64)                 
            dist.send(my_log_prob, dst = paired_rank)

            is_there_interchange = torch.zeros(1, device="cpu")                        
            dist.recv(is_there_interchange, src=paired_rank)                    

            
            if is_there_interchange:      # let's interchage!
                
                paired_v = torch.zeros(walker.num_spins, device="cpu", dtype = torch.float32)                                
                dist.recv(paired_v, src=paired_rank)    
                v = walker.v.cpu()
                dist.send(v, dst = paired_rank)                  
                walker.v = paired_v.to(walker.device)
                walker.ie = pe -walker.e_min
                walker.iq = pq -walker.q_min
                
    
