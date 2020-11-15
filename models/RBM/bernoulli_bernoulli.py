import torch
import time
import copy

class RBM(torch.nn.Module):

    def __init__(self,
                 num_visible=784,
                 num_hidden=500,
                 num_sampling_iter=2,
                 param_path = 'parameters/RBM0_best_param.pt'):
        
        # run nn.Module's constructor
        
        super().__init__()
        
        # where to save parameters
        
        self.param_path = param_path
        
        # assign properties
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_sampling_iter = num_sampling_iter
        
        # initialize W matrix and register it as a parameter
        
        W = torch.empty(num_hidden,num_visible)
        torch.nn.init.xavier_uniform_(W, gain=1.0)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.W = torch.nn.Parameter(W,requires_grad = True)
        
        # initialize bias vectors and register them as parameters
        
        b = torch.empty(num_visible,1)
        torch.nn.init.xavier_uniform_(b, gain=1.0)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.b = torch.nn.Parameter(b,requires_grad = True)
        
        c = torch.empty(num_hidden,1)
        torch.nn.init.xavier_uniform_(c, gain=1.0)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.c = torch.nn.Parameter(c,requires_grad = True)

    def sample_h_given_v(self,v):
        """
        given a visible vector v, sample a binary random vector h
        """
        p_h_given_v = torch.sigmoid(torch.matmul(self.W,v) + self.c)
        h_given_v = torch.bernoulli(p_h_given_v).detach()
        return h_given_v

    def sample_v_given_h(self,h):
        p_v_given_h = torch.sigmoid(torch.matmul(self.W.transpose(-2,-1),h) + 
                                                 self.b)
        v_given_h = torch.bernoulli(p_v_given_h).detach()
        return v_given_h
    
    def energy_func(self,v,h):
        """
        E(v,h) = -b^T v - c^T h - h^T W v
        """
        
        first_term = -torch.matmul(self.b.transpose(-2,-1),v).squeeze()
        second_term = -torch.matmul(self.c.transpose(-2,-1),h).squeeze()
        third_term = -torch.matmul(h.transpose(-2,-1),
                                   torch.matmul(self.W,v)).squeeze()
        
        energy = first_term + second_term + third_term
        
        return energy
    
    # def free_energy(self,v):
    #     """
    #     Given binary visible units, the free energy is:
        
    #     F(v) = -b^{T}v - sum_{i} log(1 + exp(c_i + W_{i}v))
        
    #     See equation 9 here for details:
            
    #     http://deeplearning.net/tutorial/rbm.html
        
    #     """
    #     first_term = -torch.matmul(self.b.transpose(-2,-1),v).squeeze()
    #     second_term = torch.sum(
    #                   torch.log(1.0 +
    #                             torch.exp(torch.matmul(self.W,v) + self.c)),
    #                   dim = 1).squeeze()
    #     return -first_term - second_term
    
    def forward(self,v0):
        
        # sample hidden value based on training example according to CD
        
        h_given_v = self.sample_h_given_v(v0)
        h_given_v0 = h_given_v.clone().detach()
        
        # start Gibbs sampling
        
        for _ in range(self.num_sampling_iter):
            v_given_h = self.sample_v_given_h(h_given_v)
            h_given_v = self.sample_h_given_v(v_given_h)
        
        # final samples from the joint distribution p(v,h)
        
        v = v_given_h
        h = h_given_v
        
        return v,h,h_given_v0
    
    def train_batch(self,x,optimizer,device):
    
        # move to GPU if available
        
        x = x.to(device)
          
        # sample batches from p(h|v) and p(v,h)
    
        v,h,h_given_v = self.__call__(x)
        
        # compute energy of batch for first term in gradient of
        # log-likelihood function log(p(v))
        
        energy1 = self.energy_func(v,h)
        
        # sample mean of gradient of energy function is equal to gradient of
        # sample mean of energy function. Note that the same is not necessarily
        # true for expectations
        
        mean_energy1 = torch.mean(energy1)
        
        # compute energy of batch for second term in gradient of
        # log-likelihood function log(p(v))
        
        energy2 = self.energy_func(x,h_given_v)
        
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
        
        # update the RBM parameters
        
        optimizer.step()
        
        # zero the accumulated parameter gradients
            
        optimizer.zero_grad()
        
        return mse
    
    def train_epoch(self,dataloader,optimizer,device,previous_RBMs):
    
        print('Training...')
        
        # to compute total training MSE reconstruction loss over entire epoch
        
        total_mse_loss = 0
        
        for i,x in enumerate(dataloader):
            
            # track progress
            
            print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                             len(dataloader.dataset)*100),
                  end='',flush=True)
            
            # if training as part of a deep belief network, sample a batch
            # of hidden vectors from previous RBMs starting from the first
            # one and use that instead as an input
            
            if previous_RBMs is not None:
                for RBM in previous_RBMs:
                    x = RBM(x)[1]
            
            # train over the batch
            
            mse_loss = self.train_batch(x,optimizer,device)
            
            # record total loss
            
            total_mse_loss += mse_loss
        
        # average training loss per sample
        
        average_mse_loss = total_mse_loss / len(dataloader.dataset)
        
        return average_mse_loss
    
    def val_batch(self,x,device):
    
        with torch.no_grad():
            
            # move to GPU if available
            
            x = x.to(device)
              
            # sample batches from p(h|v) and p(v,h)
            
            v,h,h_given_v = self.__call__(x)
            
            # validation MSE loss
            
            mse = torch.mean((x - v)**2).item()
        
        return mse
    
    def val_epoch(self,dataloader,device,previous_RBMs):

        print('\nValidating...')
        
        # to compute total validation loss over entire epoch
        
        total_mse_loss = 0
        
        for i,x in enumerate(dataloader):
            
            # track progress
            
            print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                             len(dataloader.dataset)*100),
                  end='',flush=True)
            
            # if training as part of a deep belief network, sample a batch
            # of hidden vectors from previous RBMs starting from the first
            # one and use that instead as an input
            
            if previous_RBMs is not None:
                for RBM in previous_RBMs:
                    x = RBM(x)[1]
            
            # validate over the batch
            
            mse_loss = self.val_batch(x,device)
            
            # record total loss
            
            total_mse_loss += mse_loss
        
        # average validation loss per sample
        
        average_mse_loss = total_mse_loss / len(dataloader.dataset)
        
        return average_mse_loss
    
    def train_and_val(self,num_epochs,dataloaders,optimizer,device,
                      previous_RBMs = None):
        
        # record the best loss across epochs
        
        best_val_loss = 1e10
        
        # starting time
    
        start = time.time()
        
        for epoch in range(num_epochs):
            
            # show number of epochs elapsed
            
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            
            # record the epoch start time
            
            epoch_start = time.time()
            
            # train for an epoch
            
            train_loss = self.train_epoch(dataloaders['train'],optimizer,
                                          device,previous_RBMs)
            
            # show results
            
            print('\nMSE: {:.5f}'.format(train_loss))
            
            # validate for an epoch
            
            val_loss = self.val_epoch(dataloaders['val'],device,
                                      previous_RBMs)
            
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
                
                best_parameters = copy.deepcopy(self.state_dict())
                
                torch.save(best_parameters,self.param_path)
        
        # show training and validation time and best validation accuracy
        
        end = time.time()
        total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
        print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
        print('Best MSE Loss: {:.5f}'.format(best_val_loss))

if __name__ == '__main__':
    
    rbm = RBM()
    