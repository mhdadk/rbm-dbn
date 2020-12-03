import torch
import time

class RBM_BB(torch.nn.Module):

    def __init__(self,
                 num_visible = 25,
                 num_hidden = 12,
                 num_sampling_iter = 2,
                 param_path = 'parameters/RBM0_best_param.pt'):
        
        # run nn.Module's constructor
        
        super().__init__()
        
        # where to save parameters
        
        self.param_path = param_path
        
        # number of iterations of Gibbs sampling/contrastive divergence
        
        self.num_sampling_iter = num_sampling_iter
        
        # initialize W matrix and register it as a parameter
        
        W = torch.empty(num_hidden,num_visible)
        torch.nn.init.normal_(W, mean = 0.0, std = 0.01)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.W = torch.nn.Parameter(W,requires_grad = True)
        
        # initialize bias vectors and register them as parameters
        
        b = torch.empty(num_visible,1)
        torch.nn.init.normal_(b, mean = 0.0, std = 0.01)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.b = torch.nn.Parameter(b,requires_grad = True)
        
        c = torch.empty(num_hidden,1)
        torch.nn.init.normal_(c, mean = 0.0, std = 0.01)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.c = torch.nn.Parameter(c,requires_grad = True)

    def compute_p_h_given_v(self,v):
        return torch.sigmoid(torch.matmul(self.W,v) + self.c)

    def sample_h_given_v(self,v):
        p_h_given_v = self.compute_p_h_given_v(v)
        h_given_v = torch.distributions.bernoulli.Bernoulli(probs = p_h_given_v).sample()
        return h_given_v.detach()
    
    def compute_p_v_given_h(self,h):
        return torch.sigmoid(torch.matmul(self.W.T,h) + self.b)
    
    def sample_v_given_h(self,h):
        p_v_given_h = self.compute_p_v_given_h(h)
        v_given_h = torch.distributions.bernoulli.Bernoulli(probs = p_v_given_h).sample()
        return v_given_h.detach()
    
    def energy_func(self,v,h):     
        first_term = torch.matmul(self.b.T,v)
        second_term = torch.matmul(self.c.T,h)
        third_term = torch.matmul(v.transpose(2,1),
                                  torch.matmul(self.W.T,h))
        
        # batch of energies
        
        energy = -first_term - second_term - third_term
        
        # remove redundant dimensions
        
        return energy.squeeze()
        
    def forward(self,v_0):
        
        # sample hidden vector based on training example according to
        # contrastive divergence
        
        h_given_v = self.sample_h_given_v(v_0)
        
        # need this for gradient of log-likelihood function
        
        h_given_v_0 = h_given_v.clone().detach()
        
        # Gibbs sampling
        
        for _ in range(self.num_sampling_iter):
            v_given_h = self.sample_v_given_h(h_given_v)
            h_given_v = self.sample_h_given_v(v_given_h)
        
        # return v,h from p(v,h) and h|v from p(h|v)
        
        return v_given_h,h_given_v,h_given_v_0
    
    def train_batch(self,x,optimizer,device):
    
        # move to GPU if available
        
        x = x.to(device)
          
        # sample batches from p(v,h) and p(h|v) 
    
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
        
        # update the RBM parameters using these gradients
        
        optimizer.step()
        
        # zero the accumulated parameter gradients for the next iteration
            
        optimizer.zero_grad()
        
        # compute batch reconstruction accuracy
        
        acc = torch.mean((x == v).to(torch.float)).item()
        
        return acc
    
    def val_batch(self,x,device):
    
        with torch.no_grad():
            
            # move to GPU if available
            
            x = x.to(device)
              
            # sample from p(v,h)
            
            v = self.__call__(x)[0]
            
            # compute batch reconstruction accuracy
        
            acc = torch.mean((x == v).to(torch.float)).item()
        
        return acc
    
    def run_epoch(self,mode,dataloader,optimizer,device,previous_RBMs):
        
        if mode == 'train':
            print('Training...')
        else:
            print('\nValidating...')
        
        # to compute average reconstruction accuracy per epoch
        
        recon_acc = 0
        
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
            
            # train/val over the batch and record batch reconstruction
            # accuracy
            
            if mode == 'train':
                recon_acc += self.train_batch(x,optimizer,device)
            else:
                recon_acc += self.val_batch(x,device)
        
        # average reconstruction accuracy per sample
        
        mean_recon_acc = recon_acc / len(dataloader.dataset)
        
        return mean_recon_acc
    
    def train_and_val(self,num_epochs,dataloaders,optimizer,device,
                      previous_RBMs = None):
        
        # record the best validation accuracy
        
        best_val_acc = 0
        
        # starting time
    
        start = time.time()
        
        for epoch in range(num_epochs):
            
            # show number of epochs elapsed
            
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            
            # record the epoch start time
            
            epoch_start = time.time()
            
            # train for an epoch
        
            train_acc = self.run_epoch(mode = 'train',
                                       dataloader = dataloaders['train'],
                                       optimizer = optimizer,
                                       device = device,
                                       previous_RBMs = previous_RBMs)
            
            # show results
            
            print('\nTraining Accuracy: {:.2f}%'.format(train_acc*100))
            
            # validate for an epoch
            
            val_acc = self.run_epoch(mode = 'val',
                                     dataloader = dataloaders['val'],
                                     optimizer = optimizer,
                                     device = device,
                                     previous_RBMs = previous_RBMs)
            
            # show results
            
            print('\nValidation Accuracy: {:.2f}%'.format(val_acc*100))
            
            # update learning rate
            
            # scheduler.step()
            
            # show epoch time
            
            epoch_end = time.time()
            
            epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
            
            print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
            
            # save the weights for the best validation accuracy
        
            if val_acc > best_val_acc:
                print('Saving checkpoint...')
                best_val_acc = val_acc
                torch.save(self.state_dict(),self.param_path)
        
        # show training and validation time and best validation accuracy
        
        end = time.time()
        total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
        print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
        print('Best Validation Accuracy: {:.2f}'.format(best_val_acc*100))

if __name__ == '__main__':
    
    torch.manual_seed(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    batch_size = 8
    num_visible = 25
    num_hidden = 12
    num_sampling_iter = 2
    
    rbm = RBM_BB(num_visible = num_visible,
                 num_hidden = num_hidden,
                 num_sampling_iter = num_sampling_iter).to(device)
    
    x = torch.randn(batch_size,num_visible,1).to(device)
    
    # test 1
    
    v,h,h_given_V0 = rbm(x)
    
    # test 2
    
    energy = rbm.energy_func(v,h)
    
    # test 3
    
    optimizer = torch.optim.SGD(params = rbm.parameters(),
                                lr = 0.01/batch_size,
                                momentum = 0.9,
                                weight_decay = 0.001)
    
    train_acc = rbm.train_batch(x,optimizer,device)
    print('\nTraining Accuracy: {:.2f}%'.format(train_acc*100))
    
    # test 4
    
    val_acc = rbm.val_batch(x,device)
    print('\nValidation Accuracy: {:.2f}%'.format(val_acc*100))
    