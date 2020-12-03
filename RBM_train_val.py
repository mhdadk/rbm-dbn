import torch

from torch_datasets.CSVDataset import CSVDataset
from models.RBM.categorical_bernoulli import RBM_CB

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize RBM

rbm = RBM_CB(num_visible = 50,
             num_hidden = 50,
             num_categories = 5,
             num_sampling_iter = 3).to(device)

# initialize dataloaders

train_path = 'data/train2.csv'
val_path = 'data/val2.csv'
dataloaders = {}

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# batch sizes for training and validation

train_batch_size = 128
val_batch_size = 128

for mode,path,batch_size in [('train',train_path,train_batch_size),
                             ('val',val_path,val_batch_size)]:
    
    dataset = CSVDataset(path)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               **dl_config)

optimizer = torch.optim.SGD(params = rbm.parameters(),
                            lr = 0.01,#/train_batch_size,
                            momentum = 0.9,
                            weight_decay = 0)

# number of epochs to train and validate for

num_epochs = 10

"""
need to add this if statement such that the multiprocessing package
used by the PyTorch dataloaders works correctly. See here for details:

https://discuss.pytorch.org/t/if-name-main-for-window10/19377    

https://stackoverflow.com/questions/20222534/python-multiprocessing-on-windows-if-name-main

Also, this code MUST be run on the command line and NOT through an
IPython Notebook so that it doesn't throw an error.

"""

if __name__ == '__main__':
    rbm.train_and_val(num_epochs,dataloaders,optimizer,device)
