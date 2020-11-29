import torch

batch_size = 32
num_visible = 50
num_hidden = 25
num_categories = 5

v = torch.zeros((batch_size,num_categories,num_visible,1))
h1 = torch.randn(batch_size,num_hidden,1)

W = torch.randn(num_categories,num_hidden,num_visible)
b = torch.randn(num_categories,num_visible,1)
c = torch.randn(num_hidden,1)

def compute_p_v_given_h(h,W,b):
    linears = []
    for i in range(num_categories):
        linear = torch.matmul(W[i].transpose(-2,-1),h) + b[i]
        linears.append(linear.squeeze(dim=-1))
    log_p_v_given_h = torch.stack(linears,dim = 2)
    p_v_given_h = torch.nn.functional.softmax(log_p_v_given_h, dim = 2)
    return p_v_given_h

def compute_p_h_given_v(v,W,c):
    linear = torch.sum(torch.matmul(W,v),dim = 1) + c
    p_h_given_v = torch.sigmoid(linear)
    return p_h_given_v

p_v_given_h = compute_p_v_given_h(h1,W,b)

v_given_h = torch.zeros_like(p_v_given_h)

for i,batch in enumerate(p_v_given_h):
    dist = torch.distributions.one_hot_categorical.OneHotCategorical(
            probs = batch)
    v_given_h[i] = dist.sample()

p_h_given_v = compute_p_h_given_v(v,W,c)

h_given_v = torch.zeros_like(p_h_given_v)

for i,batch in enumerate(p_h_given_v):
    dist = torch.distributions.bernoulli.Bernoulli(probs = batch)
    h_given_v[i] = dist.sample()

first_term_inner = torch.matmul(W,v)

# for matrix multiplication

h1 = torch.unsqueeze(h1,dim=1)

# outer summation over num_hidden hidden units

first_term_outer = torch.matmul(h1.transpose(-2,-1),first_term_inner).squeeze()

# undo after

h1 = torch.squeeze(h1,dim=1)

# outer outer summation over num_categories classes. This will return a
# batch of first terms

first_term = -torch.sum(first_term_outer, dim = -1)

# inner summation over num_visible visible units

second_term_inner = torch.matmul(v.transpose(-2,-1),b).squeeze()

# outer summation over num_categories classes. This will return a
# batch of second terms

second_term = -torch.sum(second_term_inner, dim = -1)

# this will return a batch of third terms

third_term = -torch.matmul(h1.transpose(-2,-1),c).squeeze()

# this is a batch of energies

energy = first_term + second_term + third_term