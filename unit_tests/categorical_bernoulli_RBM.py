import torch

batch_size = 32
num_visible = 50
num_hidden = 25
num_categories = 5

# create matrix of one-hot encoded categorical random variables. One-hot
# encoding is necessary to avoid biasing computations that involve these
# categorical random variables

# need the last dimension in v to broadcast matrix multiplication

v = torch.zeros((batch_size,num_categories,num_visible,1))
# idx = torch.randint(0,5,(num_visible,))
# for i in range(num_visible):
#     v[:,idx[i],i,:] = 1
h1 = torch.randn(batch_size,num_hidden,1)

"""
             _____
            |     |
            |  h  |
            |_____|
           /   |   \
          /    |    \   ...
         /     |     \
  _____      _____     _____
 |     |    |     |   |     |
 | v_1 |    | v_2 |   | v_3 |    ...
 |_____|    |_____|   |_____|

The diagram above shows the connections between one Bernoulli hidden unit
and the corresponding categorical visible unit. Since the categorical
visible unit can be expressed in one-hot encoding format, then it can
be expressed as three separate visible units v_1, v_2, and v_3.

Each connection is associated with a weight w and bias b, such that, for
example, v_1 is a function of w_1,b_1, and h. Now consider the
above configuration for all hidden units and visible units. The hidden
unit h becomes a vector of hidden variables h, while the visible units
become a matrix of visible vectors V, such that each column of this matrix
represents a column vector containing the one-hot encoding of a categorical
visible unit.

To generalize the weights w and biases b for all visible and hidden
units, notice that for a single hidden unit connected to the corresponding
Bernoulli visible units, the weights can be represented by a vector. To
go from the scalar h to the vector v, the vector of weights can be scaled
as follows:
    
    v = h * [w_1,w_2,...,w_K]
    
Since for all visible units, v becomes the matrix V, h becomes a vector,
and the weights are all contained in a matrix, then obtaining a matrix
V corresponds to broadcasting.
"""

W = torch.randn(num_categories,num_hidden,num_visible)
b = torch.randn(num_categories,num_visible,1)
c = torch.randn(num_hidden,1)

"""
since p(v|h) is a multinomial distribution, where v is a vector, then
it is described by a num_categories x num_visible matrix, where the i^th
column is the categorical probability distribution for the i^th element
of v.

Since p(v|h) is a num_categories x num_visible matrix, then the intermediate
values in the i^th row can be computed as:
    
i^th weight matrix * h + i^th bias vector

Then, once these intermediate values are computed for all rows,
p(v|h) is computed by computing the softmax function along each column of
the p(v|h) matrix.
"""

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

#%% energy function

# inner summation over num_visible visible units

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

# first_term = -torch.matmul(b.transpose(-2,-1),v).squeeze()
# first_term = torch.sum(first_term,dim = 0)
# second_term = -torch.matmul(self.c.transpose(-2,-1),h).squeeze()
# third_term = -torch.matmul(h.transpose(-2,-1),
#                            torch.matmul(self.W,v)).squeeze()
# third_term = torch.sum(third_term,dim = 0)

# energy = first_term + second_term + third_term