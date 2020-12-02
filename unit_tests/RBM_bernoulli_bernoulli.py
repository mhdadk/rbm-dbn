import torch

batch_size = 8
num_visible = 4 # 25
num_hidden = 3 # 12

v1 = torch.randn(batch_size,num_visible,1)
h1 = torch.randn(batch_size,num_hidden,1)
W1 = torch.randn(num_hidden,num_visible)
b1 = torch.randn(num_visible,1)
c1 = torch.randn(num_hidden,1)

# p(v|h)

p_v_given_h = torch.sigmoid(torch.matmul(W1.T,h1) + b1)

# sample from p(v|h)

v_sample = torch.distributions.bernoulli.Bernoulli(probs = p_v_given_h).sample()

# p(h|v)

p_h_given_v = torch.sigmoid(torch.matmul(W1,v1) + c1)

# sample from p(h|v)

h_sample = torch.distributions.bernoulli.Bernoulli(probs = p_h_given_v).sample()

# compute energy function

first_term = torch.matmul(b1.T,v1)
second_term = torch.matmul(c1.T,h1)
third_term = torch.matmul(v1.T,torch.matmul(W1.T,h1))

# batch of energies

energy = -first_term - second_term - third_term
