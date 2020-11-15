import torch
import os

from CSVDataset import CSVDataset
from models.DBN.DBN import DBN

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize DBN

dbn = DBN(num_RBMs = 3,
          num_visible = [50,25,15],
          num_hidden = [25,15,10],
          num_categories = 5,
          num_sampling_iter = [5,5,5],
          device = device)

# initialize dataloaders

data_dir = '../../data/projG'
train_path = os.path.join(data_dir,'data_train.csv')
val_path = os.path.join(data_dir,'data_val.csv')
test_path = os.path.join(data_dir,'data_test.csv')
dataloaders = {}

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# batch sizes for training, validation, and testing

train_batch_size = 64
val_batch_size = 64
test_batch_size = 64

for mode,path,batch_size in [('train',train_path,train_batch_size),
                             ('val',val_path,val_batch_size),
                             ('test',test_path,test_batch_size)]:
    
    dataset = CSVDataset(path)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               **dl_config)

optimizer1 = torch.optim.SGD(params = dbn.rbms[0].parameters(),
                             lr = 1,#0.01/train_batch_size,
                             momentum = 0.9,
                             weight_decay = 0.001)

optimizer2 = torch.optim.SGD(params = dbn.rbms[1].parameters(),
                             lr = 1,#0.01/train_batch_size,
                             momentum = 0.9,
                             weight_decay = 0.001)

optimizer3 = torch.optim.SGD(params = dbn.rbms[2].parameters(),
                             lr = 1,#0.01/train_batch_size,
                             momentum = 0.9,
                             weight_decay = 0.001)

optimizers = [optimizer1,optimizer2,optimizer3]

# number of epochs to train and validate each RBM for

num_epochs = [5,5,5]

"""
need to add this if statement such that the multiprocessing package
used by the PyTorch dataloaders works correctly. See here for details:

https://discuss.pytorch.org/t/if-name-main-for-window10/19377    

https://stackoverflow.com/questions/20222534/python-multiprocessing-on-windows-if-name-main

Also, this code MUST be run on the command line and NOT through an
IPython Notebook so that it doesn't throw an error.

"""

if __name__ == '__main__':
    
    dbn.train(num_epochs = num_epochs,
              dataloaders = dataloaders,
              optimizers = optimizers,
              device = device)
    
    # test example
    
    x = torch.randn(1,5,50,1).to(device)
    y = dbn(x)
        