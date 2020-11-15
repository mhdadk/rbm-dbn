import torch
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import random

from bernoulli_bernoulli import RBM

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = False#torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

test_dataset = tv.datasets.MNIST(root = '.',
                                 train = False,
                                 download = True,
                                 transform = tv.transforms.ToTensor())

rbm = RBM().to(device)

# load best parameters
    
rbm.load_state_dict(torch.load('best_param.pt'))

"""
need to add this if statement such that the multiprocessing package
used by the PyTorch dataloaders works correctly. See here for details:

https://discuss.pytorch.org/t/if-name-main-for-window10/19377    

https://stackoverflow.com/questions/20222534/python-multiprocessing-on-windows-if-name-main

Also, this code MUST be run on the command line and NOT through an
IPython Notebook so that it doesn't throw an error.

"""

if __name__ == '__main__': 

    # randomly sample a batch
    
    idx = random.randint(0,len(test_dataset)-1)   
    x,_ = test_dataset[idx]
    
    # MNIST image values range from 0 to 1, so need to threshold
    
    x = (x > 0).type(torch.float)
    
    # move to GPU if available
    
    x = x.to(device)
    
    # flatten to batch of 784-dimensional row vectors
    
    x = torch.flatten(x,-2,-1)
    
    # convert batch of row vectors to batch of column vectors
    
    x = torch.transpose(x,-2,-1).unsqueeze(dim=0)
    
    # generate a reconstruction
    
    v,_,_ = rbm(x)
    
    # convert to image
    
    a = x[0].reshape((28,28)).numpy().astype(np.uint8) * 255
    b = v[0].reshape((28,28)).numpy().astype(np.uint8) * 255
    
    # show images
    
    plt.subplot(2,1,1)
    plt.imshow(a)
    plt.subplot(2,1,2)
    plt.imshow(b)