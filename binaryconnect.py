import torch.nn as nn
import numpy
from torch.autograd import Variable

'''
 This class implements the BinaryConnect (BC) method,
  which binaries the weights of a neural network to -1 or +1 during the forward and backward passes
'''
class BC():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        #self.target_params = []
        self.target_modules = []
        
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        # Binarizes the weights of the target layers to -1 or +1.
        self.save_params()
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign())
            
            
    def BWN(self): # for Binary Weight Network
        self.save_params()
        for index in range(self.num_of_params):
            E=self.target_modules[index].data.abs().mean()
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign() *E)
            
            
    def save_params(self):
        # Requires real-valued weights to accurately update the model parameters using the computed gradients.
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)


    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        # clip the real-valued weights within the [−1, 1] interval right after the weight updates
        clip_scale=[]
        m=nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)
