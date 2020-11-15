import torch
import time
import copy
import os

from CSVDataset import CSVDataset
from models.RBM.categorical_bernoulli import RBM

# helper functions

def train_batch(x,rbm,optimizer,device):
    
    # move to GPU if available
    
    x = x.to(device)
      
    # sample batches from p(h|v) and p(v,h)

    v,h,h_given_v = rbm(x)
    
    # compute energy of batch for first term in gradient of
    # log-likelihood function log(p(v))
    
    energy1 = rbm.energy_func(v,h)
    
    # sample mean of gradient of energy function is equal to gradient of
    # sample mean of energy function. Note that the same is not necessarily
    # true for expectations
    
    mean_energy1 = torch.mean(energy1)
    
    # compute energy of batch for second term in gradient of
    # log-likelihood function log(p(v))
    
    energy2 = rbm.energy_func(x,h_given_v)
    
    # sample mean of gradient of energy function is equal to gradient of
    # sample mean of energy function. Note that the same is not necessarily
    # true for expectations
    
    mean_energy2 = torch.mean(energy2)
    
    # total mean energy
    
    total_mean_energy = mean_energy1 - mean_energy2
    
    # since PyTorch can only perform gradient descent, then gradient
    # ascent is equivalent to performing gradient descent on the negative
    # of total_mean_energy
    
    total_mean_energy *= -1
    
    # compute gradient of log-likelihood with respect to parameters
    
    total_mean_energy.backward()
    
    # compute training MSE reconstruction error
    
    mse = torch.mean((x - v)**2).item()
    
    optimizer.step()
    
    # zero the accumulated parameter gradients
        
    optimizer.zero_grad()
    
    return mse

def train_epoch(rbm,dataloader,optimizer,device):
    
    print('Training...')
    
    # to compute total training MSE reconstruction loss over entire epoch
    
    total_mse_loss = 0
    
    for i,x in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # train over the batch
        
        mse_loss = train_batch(x,rbm,optimizer,device)
        
        # record total loss
        
        total_mse_loss += mse_loss
    
    # average training loss per sample
    
    average_mse_loss = total_mse_loss / len(dataloader.dataset)
    
    return average_mse_loss

def val_batch(x,rbm,device):
    
    with torch.no_grad():
        
        # move to GPU if available
        
        x = x.to(device)
          
        # sample batches from p(h|v) and p(v,h)
    
        v,h,h_given_v = rbm(x)
        
        # validation MSE loss
        
        mse = torch.mean((x - v)**2).item()
    
    return mse

def val_epoch(rbm,dataloader,device):

    print('\nValidating...')
    
    # to compute total validation loss over entire epoch
    
    total_mse_loss = 0
    
    for i,x in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # validate over the batch
        
        mse_loss = val_batch(x,rbm,device)
        
        # record total loss
        
        total_mse_loss += mse_loss
    
    # average validation loss per sample
    
    average_mse_loss = total_mse_loss / len(dataloader.dataset)
    
    return average_mse_loss

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize RBM

rbm = RBM(num_visible = 50, num_hidden = 25, num_categories = 5,
          num_sampling_iter = 5).to(device)

# initialize dataloaders

data_dir = '../../data/projG'
train_path = os.path.join(data_dir,'data_train.csv')
val_path = os.path.join(data_dir,'data_val.csv')
test_path = os.path.join(data_dir,'data_test.csv')
dataloaders = {}

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

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
                            lr = 0.01/train_batch_size,
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
        
        train_loss = train_batch(x,rbm,optimizer,device)
        
        # show results
        
        print('MSE: {:.5f}'.format(train_loss))
        
        # validation ######################################################
        
        print('\nValidating...')
        
        # sample a batch
        
        x = next(iter(dataloaders['val']))
        
        # validate over the batch
        
        val_loss = val_batch(x,rbm,device)
        
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
        
        test_loss = val_batch(x,rbm,device)
        
        # show results
        
        print('MSE: {:.5f}'.format(test_loss))
        
    else:
        
        for epoch in range(num_epochs):
            
            # show number of epochs elapsed
            
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            
            # record the epoch start time
            
            epoch_start = time.time()
            
            # train for an epoch
        
            train_loss = train_epoch(rbm,dataloaders['train'],optimizer,
                                     device)
            
            # show results
            
            print('\nMSE: {:.5f}'.format(train_loss))
            
            # validate for an epoch
            
            val_loss = val_epoch(rbm,dataloaders['val'],device)
            
            # show results
            
            print('\nMSE: {:.5f}'.format(val_loss))
            
            # update learning rate
            
            # scheduler.step()
            
            # show epoch time
            
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
        
        # show training and validation time and best validation accuracy
        
        end = time.time()
        total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
        print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
        print('Best MSE Loss: {:.5f}'.format(best_val_loss))
