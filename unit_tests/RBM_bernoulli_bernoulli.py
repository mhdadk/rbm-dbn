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

p_v_given_h = torch.sigmoid(torch.matmul(W1.T,h1) + b1.unsqueez)

# sample from p(v|h)

v_sample = torch.distributions.bernoulli.Bernoulli(probs = p_v_given_h).sample()

# p(h|v)

WV = torch.multiply(W1.permute(2,0,1),V1.unsqueeze(1)) # unsqueeze needed for broadcasting
q = torch.sum(WV,dim = (2,3)) + c1
p_h_given_V = torch.sigmoid(q)

# sample from p(h|V)

h_sample = torch.distributions.bernoulli.Bernoulli(probs = p_h_given_V).sample()

# compute energy function

# first term

Wh = torch.multiply(W1,h1.unsqueeze(1).unsqueeze(1))
WhV = torch.multiply(Wh.permute(0,3,1,2),V1.unsqueeze(1))
first_term = torch.sum(WhV,dim = (1,2,3))

# second term

VB = torch.multiply(V1,B1)
second_term = torch.sum(VB,dim = (1,2))

# third term

third_term = torch.matmul(h1,c1)

# batch of energies

energy = -first_term - second_term - third_term
