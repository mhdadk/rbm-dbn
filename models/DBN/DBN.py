import torch

from .categorical_bernoulli import RBM as RBM_cb
from .bernoulli_bernoulli import RBM as RBM_bb

class DBN(torch.nn.Module):

    def __init__(self,
                 num_RBMs = 2,
                 num_visible = [50,25],
                 num_hidden = [25,12],
                 num_categories = 5,
                 num_sampling_iter = [2,2],
                 device = 'cuda'):
        
        # run nn.Module's constructor
        
        super().__init__()
        
        # list of RBMs
        
        self.rbms = []
        
        # first RBM is always a categorical-Bernoulli RBM
        
        param_path = 'parameters/RBM0_param.pt'
        
        self.rbms.append(RBM_cb(num_visible = num_visible[0],
                                num_hidden = num_hidden[0],
                                num_categories = num_categories,
                                num_sampling_iter = num_sampling_iter[0],
                                param_path = param_path).to(device))
        
        # append the other RBMs
        
        for i,(num_visible,num_hidden,num_sampling_iter) in enumerate(zip(num_visible[1:],
                                                                          num_hidden[1:],
                                                                          num_sampling_iter[1:])):
            
            param_path = 'parameters/RBM'+str(i+1)+'_param.pt'
            
            self.rbms.append(RBM_bb(num_visible = num_visible,
                                    num_hidden = num_hidden,
                                    num_sampling_iter = num_sampling_iter,
                                    param_path = param_path).to(device))
    
    def train(self,num_epochs,dataloaders,optimizers,device):
        
        # train and validate first/root rbm only
        
        print("\nTraining and validating RBM number 1...")
        
        self.rbms[0].train_and_val(num_epochs[0],dataloaders,optimizers[0],
                                   device)
        
        # train and validate the subsequent RBMs in order
        
        for i in range(1,len(self.rbms)):
            
            print("\nTraining and validating RBM number {}...".format(i+1))
            
            self.rbms[i].train_and_val(num_epochs[i],
                                       dataloaders,
                                       optimizers[i],
                                       device,
                                       self.rbms[:i])
    
    def forward(self,x):
        
        # sample hidden vector from final hidden layer
        
        for RBM in self.rbms:
            x = RBM(x)[1]
            
        return x
