import torch
import time
import copy
import os

from CSVDataset import CSVDataset
from models.RBM.categorical_bernoulli import RBM

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize RBM

rbm = RBM(num_visible = 50, num_hidden = 50, num_categories = 5,
          num_sampling_iter = 5).to(device)

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

optimizer = torch.optim.SGD(params = rbm.parameters(),
                            lr = 1,#0.01/train_batch_size,
                            momentum = 0.9,
                            weight_decay = 0.001)

# whether to sample a single batch for a trial run
    
trial_run = False

# otherwise, set the number of epochs to train and validate for

if not trial_run:
    num_epochs = 10

# record the best loss across epochs

best_val_loss = 1e10

"""
need to add this if statement such that the multiprocessing package
used by the PyTorch dataloaders works correctly. See here for details:

https://discuss.pytorch.org/t/if-name-main-for-window10/19377    

https://stackoverflow.com/questions/20222534/python-multiprocessing-on-windows-if-name-main

Also, this code MUST be run on the command line and NOT through an
IPython Notebook so that it doesn't throw an error.

"""

if __name__ == '__main__':

    # starting time
    
    start = time.time()
    
    if trial_run:
            
        # record the epoch start time
        
        epoch_start = time.time()
        
        # training #######################################################
        
        print('\nTraining...')
        
        # sample a batch
        
        x = next(iter(dataloaders['train']))
        
        # train over the batch
        
        train_loss = rbm.train_batch(x,optimizer,device)
        
        # show results
        
        print('MSE: {:.5f}'.format(train_loss))
        
        # validation ######################################################
        
        print('\nValidating...')
        
        # sample a batch
        
        x = next(iter(dataloaders['val']))
        
        # validate over the batch
        
        val_loss = rbm.val_batch(x,device)
        
        # show results
        
        print('MSE: {:.5f}'.format(val_loss))
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
        print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        
        # save the weights for the best validation loss
        
        if val_loss < best_val_loss:
            
            print('Saving checkpoint...')
            
            best_val_loss = val_loss
            
            # deepcopy needed because a dict is a mutable object
            
            best_parameters = copy.deepcopy(rbm.state_dict())
            
            torch.save(best_parameters,'best_param.pt')
        
        # testing ############################################################
        
        print('\nTesting...')
        
        # load best parameters
    
        rbm.load_state_dict(torch.load('best_param.pt'))
        
        # sample a batch
        
        x = next(iter(dataloaders['test']))
        
        # test the batch
        
        test_loss = rbm.val_batch(x,device)
        
        # show results
        
        print('MSE: {:.5f}'.format(test_loss))
        
    else:
        
        rbm.train_and_val(num_epochs,dataloaders,optimizer,device)
        