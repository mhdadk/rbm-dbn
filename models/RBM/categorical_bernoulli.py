import torch
import time

class RBM_CB(torch.nn.Module):

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
        
        # number of iterations of contrastive divergence
        
        self.num_sampling_iter = num_sampling_iter
        
        # number of classes
        
        self.num_categories = num_categories
        
        # initialize W matrix and register it as a parameter
        
        W = torch.empty(num_visible,num_categories,num_hidden)
        torch.nn.init.normal_(W, mean = 0.0, std = 0.01)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.W = torch.nn.Parameter(W,requires_grad = True)
        
        # initialize bias matrix B and register it as a parameters
        
        B = torch.empty(num_visible,num_categories)
        torch.nn.init.normal_(B, mean = 0.0, std = 0.01)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.B = torch.nn.Parameter(B,requires_grad = True)
        
        # initialize bias vector c and register it as a parameter
        
        c = torch.empty(num_hidden)
        torch.nn.init.normal_(c, mean = 0.0, std = 0.01)
        # turn on requires_grad after initialization to avoid including
        # in grad_fn
        self.c = torch.nn.Parameter(c,requires_grad = True)
    
    def compute_p_V_given_h(self,h):
        # unsqueeze needed for broadcasting
        Wh = torch.multiply(self.W, h.unsqueeze(1).unsqueeze(1))
        Y = torch.sum(Wh,dim = 3) + self.B
        p_V_given_h = torch.nn.functional.softmax(Y,dim = 2)
        return p_V_given_h
    
    def sample_V_given_h(self,h):
        p_V_given_h = self.compute_p_V_given_h(h)
        V_given_h = torch.distributions.multinomial.Multinomial(total_count = 1,
                                                                probs = p_V_given_h).sample()
        return V_given_h.detach()
    
    def compute_p_h_given_V(self,V):
        # unsqueeze needed for broadcasting
        WV = torch.multiply(self.W.permute(2,0,1),V.unsqueeze(1))
        q = torch.sum(WV,dim = (2,3)) + self.c
        p_h_given_V = torch.sigmoid(q)
        return p_h_given_V
    
    def sample_h_given_V(self,V):
        p_h_given_V = self.compute_p_h_given_V(V)
        h_given_V = torch.distributions.bernoulli.Bernoulli(probs = p_h_given_V).sample()
        return h_given_V.detach()

    def energy_func(self,V,h):
        
        # first term
        
        Wh = torch.multiply(self.W,h.unsqueeze(1).unsqueeze(1))
        WhV = torch.multiply(Wh.permute(0,3,1,2),V.unsqueeze(1))
        first_term = torch.sum(WhV,dim = (1,2,3))
        
        # second term
        
        VB = torch.multiply(V,self.B)
        second_term = torch.sum(VB,dim = (1,2))
        
        # third term
        
        third_term = torch.matmul(h,self.c)
        
        # batch of energies
        
        energy = -first_term - second_term - third_term
        
        return energy
    
    def forward(self,V_0):
        
        # sample hidden vector based on training example according to
        # contrastive divergence
        
        h_given_V = self.sample_h_given_V(V_0)
        
        # need this for gradient of log-likelihood function
        
        h_given_V_0 = h_given_V.clone().detach()
        
        # Gibbs sampling
        
        for _ in range(self.num_sampling_iter):
            V_given_h = self.sample_V_given_h(h_given_V)
            h_given_V = self.sample_h_given_V(V_given_h)
        
        # return V,h from p(V,h) and h|V from p(h|V)
        
        return V_given_h,h_given_V,h_given_V_0
    
    def train_batch(self,x,optimizer,device):
    
        # move to GPU if available
        
        x = x.to(device)
          
        # sample batches from p(V,h) and p(h|V) 
    
        V,h,h_given_V = self.__call__(x)
        
        # compute batch of energies for first term in gradient of
        # log-likelihood function log(p(v))
        
        energy1 = self.energy_func(V,h)
        
        # sample mean of gradient of energy function is equal to gradient of
        # sample mean of energy function. Note that the same is not necessarily
        # true for expectations
        
        mean_energy1 = torch.mean(energy1)
        
        # compute batch of energies for second term in gradient of
        # log-likelihood function log(p(v))
        
        energy2 = self.energy_func(x,h_given_V)
        
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
        
        # compute gradients of log-likelihood with respect to parameters
        
        total_mean_energy.backward()
        
        # update the RBM parameters using these gradients
        
        optimizer.step()
        
        # zero the accumulated parameter gradients for the next iteration
            
        optimizer.zero_grad()
        
        # compute batch reconstruction accuracy using mean-field
        # propagation
        
        p_h_given_V = self.compute_p_h_given_V(V)
        p_V_given_h = self.compute_p_V_given_h(p_h_given_V)
        acc = torch.mean(torch.eq(x.argmax(dim=2),
                                  p_V_given_h.argmax(dim=2)).to(torch.float)).item()
        
        return acc
    
    def val_batch(self,x,device):
    
        with torch.no_grad():
            
            # move to GPU if available
            
            x = x.to(device)
              
            # sample from p(v,h)
            
            V = self.__call__(x)[0]
            
            # compute batch reconstruction accuracy using mean-field
            # propagation
        
            p_h_given_V = self.compute_p_h_given_V(V)
            p_V_given_h = self.compute_p_V_given_h(p_h_given_V)
            acc = torch.mean(torch.eq(x.argmax(dim=2),
                                      p_V_given_h.argmax(dim=2)).to(torch.float)).item()
        
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
        
        mean_recon_acc = recon_acc / len(dataloader)
        
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
    num_visible = 50,
    num_hidden = 25,
    num_categories = 5,
    num_sampling_iter = 2
    
    rbm = RBM_CB(num_visible = num_visible,
                 num_hidden = num_hidden,
                 num_categories = num_categories,
                 num_sampling_iter = num_sampling_iter).to(device)
    
    x = torch.randn(batch_size,num_visible,num_categories).to(device)
    
    # test 1
    
    V,h,h_given_V0 = rbm(x)
    
    # test 2
    
    energy = rbm.energy_func(V,h)
    
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
    