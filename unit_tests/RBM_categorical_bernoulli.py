import torch

batch_size = 8
num_visible = 4 # 50
num_categories = 5 # 5
num_hidden = 3 # 25

V1 = torch.randn(batch_size,num_visible,num_categories)
h1 = torch.randn(batch_size,num_hidden)
W1 = torch.randn(num_visible,num_categories,num_hidden)
B1 = torch.randn(num_visible,num_categories)
c1 = torch.randn(num_hidden)

# p(V|h)

Wh = torch.multiply(W1,h1.unsqueeze(1).unsqueeze(1)) # unsqueeze needed for broadcasting
Y1 = torch.sum(Wh,dim = 3) + B1
p_V_given_h = torch.nn.functional.softmax(Y1,dim = 2)

# sample from p(V|h)

V_sample = torch.distributions.multinomial.Multinomial(total_count = 1,
                                                       probs = p_V_given_h).sample()

# p(h|V)

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
