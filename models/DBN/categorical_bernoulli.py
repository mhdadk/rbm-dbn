import torch
import time
import copy

class RBM(torch.nn.Module):

    def __init__(self,
                 num_visible = 50,
                 num_hidden = 25,
                 num_categories = 5,
                 num_sampling_iter = 2,
                 param_path = 'parameters/RBM1_best_param.pt'):
        
        # run nn.Module's constructor
        
        super().__init__()
        
        # where to save parameters
        
        self.param_path = param_path
        
        # properties
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        # number of iterations of contrastive divergence
        
        self.num_sampling_iter = num_sampling_iter
        
        # number of classes
        
        self.num_categories = num_categories
        
        # initialize W matrix and register it as a parameter. Note that
        # since visible units are categorical, then there is a weight
        # vector associated with each visible unit, and equivalently a
        # weight matrix associated with all visible units in the RBM
        
        W = torch.empty(num_categories,num_hidden,num_visible)
        torch.nn.init.normal_(W, mean = 0.0, std = 0.01)
        #torch.nn.init.xavier_uniform_(W, gain=1.0)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.W = torch.nn.Parameter(W,requires_grad = True)
        
        # initialize bias vectors and register them as parameters. Note
        # that there exists a bias vector b for each categorical visible
        # unit, and equivalently a bias matrix associated with all visible
        # units in the RBM
        
        b = torch.empty(num_categories,num_visible,1)
        torch.nn.init.normal_(b, mean = 0.0, std = 0.01)
        #torch.nn.init.xavier_uniform_(b, gain=1.0)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.b = torch.nn.Parameter(b,requires_grad = True)
        
        # since the bias vector c is only associated with sampling
        # Bernoulli hidden units given categorical visible units, then
        # there is a scalar bias associated with each hidden unit, and
        # equivalently a bias vector associated with all hidden units
        # in the RBM
        
        c = torch.empty(num_hidden,1)
        torch.nn.init.normal_(c, mean = 0.0, std = 0.01)
        #torch.nn.init.xavier_uniform_(c, gain=1.0)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.c = torch.nn.Parameter(c,requires_grad = True)
    
    def compute_p_h_given_v(self,v):
        log_p_h_given_v = torch.sum(torch.matmul(self.W,v),dim = 1) + self.c
        p_h_given_v = torch.sigmoid(log_p_h_given_v)
        return p_h_given_v
    
    def sample_h_given_v(self,v):
        """
        given a visible vector v containing categorical random variables
        in one-hot encoding format, sample a hidden vector h containing
        Bernoulli random variables
        """
        p_h_given_v = self.compute_p_h_given_v(v)
        
        h_given_v = torch.zeros_like(p_h_given_v)
        
        for i,batch in enumerate(p_h_given_v):
            dist = torch.distributions.bernoulli.Bernoulli(probs = batch)
            h_given_v[i] = dist.sample()
        
        return h_given_v.detach()
    
    def compute_p_v_given_h(self,h):
        linears = []
        for i in range(self.num_categories):
            linear = torch.matmul(self.W[i].transpose(-2,-1),h) + self.b[i]
            linears.append(linear.squeeze(dim=-1))
        log_p_v_given_h = torch.stack(linears,dim = 2)
        p_v_given_h = torch.nn.functional.softmax(log_p_v_given_h, dim = 2)
        return p_v_given_h
    
    def sample_v_given_h(self,h):
        """
        given a hidden vector h containing Bernoulli random variables,
        sample a visible vector v containing categorical random variables
        in one-hot encoding format
        """
        
        # make sure this is a batch_size x num_samples x num_classes array
        
        p_v_given_h = self.compute_p_v_given_h(h)
        
        v_given_h = torch.zeros_like(p_v_given_h)
        
        for i,batch in enumerate(p_v_given_h):
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                    probs = batch)
            v_given_h[i] = dist.sample()
        
        # need to transpose for later
        
        v_given_h = torch.transpose(v_given_h,-2,-1)
        
        # need to add last dimension to broadcast matrix multiplication
        
        v_given_h = torch.unsqueeze(v_given_h,-1)
        
        return v_given_h.detach()
    
    def energy_func(self,v,h):
        
        # inner summation over num_visible visible units

        first_term_inner = torch.matmul(self.W,v)
        
        # for matrix multiplication
        
        h = torch.unsqueeze(h,dim=1)
        
        # outer summation over num_hidden hidden units
        
        first_term_outer = torch.matmul(h.transpose(-2,-1),first_term_inner).squeeze()
        
        # undo after
        
        h = torch.squeeze(h,dim=1)
        
        # outer outer summation over num_categories classes. This will return a
        # batch of first terms
        
        first_term = -torch.sum(first_term_outer, dim = -1)
        
        # inner summation over num_visible visible units
        
        second_term_inner = torch.matmul(v.transpose(-2,-1),self.b).squeeze()
        
        # outer summation over num_categories classes. This will return a
        # batch of second terms
        
        second_term = -torch.sum(second_term_inner, dim = -1)
        
        # this will return a batch of third terms
        
        third_term = -torch.matmul(h.transpose(-2,-1),self.c).squeeze()
        
        # this is a batch of energies
        
        energy = first_term + second_term + third_term
        
        return energy
    
    def forward(self,v0):
        
        # sample hidden value based on training example according to CD
        
        h_given_v = self.sample_h_given_v(v0)
        h_given_v0 = h_given_v.clone().detach()
        
        # start Gibbs sampling
        
        for _ in range(self.num_sampling_iter):
            v_given_h = self.sample_v_given_h(h_given_v).to(torch.float)
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
        
        #mse = torch.mean((x - v)**2).item()
        mse = torch.nn.functional.binary_cross_entropy(v,x,
                                                       reduction='mean')
        
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
                # need to move to GPU because RBM is already on GPU but
                # x is not
                x = x.to(device)
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
                # need to move to GPU because RBM is already on GPU but
                # x is not
                x = x.to(device)
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
    v = torch.randn(32,5,50,1)
    v_f,h_f,h_given_v0 = rbm(v)
    energy = rbm.energy_func(v_f,h_f)    
    