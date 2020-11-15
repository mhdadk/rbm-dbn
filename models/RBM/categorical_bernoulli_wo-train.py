import torch

class RBM(torch.nn.Module):

    def __init__(self,
                 num_visible = 50,
                 num_hidden = 25,
                 num_categories = 5,
                 num_sampling_iter = 2):
        
        # run nn.Module's constructor
        
        super().__init__()
        
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
        log_p_v_given_h = torch.stack(linears,dim = 1)
        p_v_given_h = torch.nn.functional.softmax(log_p_v_given_h, dim = 1)
        return p_v_given_h
    
    def sample_v_given_h(self,h):
        """
        given a hidden vector h containing Bernoulli random variables,
        sample a visible vector v containing categorical random variables
        in one-hot encoding format
        """
        
        # need to transpose to properly sample from OneHotCategorical
        
        p_v_given_h = self.compute_p_v_given_h(h).transpose(-2,-1)
        
        v_given_h = torch.zeros_like(p_v_given_h)
        
        for i,batch in enumerate(p_v_given_h):
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                    probs = batch)
            v_given_h[i] = dist.sample()
        
        # need to un-transpose for later
        
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

if __name__ == '__main__':
    
    rbm = RBM()
    v = torch.randn(32,5,50,1)
    v_f,h_f,h_given_v0 = rbm(v)
    energy = rbm.energy_func(v_f,h_f)    
    