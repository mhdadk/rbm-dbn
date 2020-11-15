import torch
import random

from CSVDataset import CSVDataset
from models.RBM.categorical_bernoulli import RBM

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize RBM

rbm = RBM(num_visible = 50, num_hidden = 25, num_categories = 5,
          num_sampling_iter = 5).to(device)

# load best parameters

rbm.load_state_dict(torch.load('best_param.pt'))

# sample a batch

data_dir = '../../data/projG/data_test.csv'
dataset = CSVDataset(data_dir)
idx = random.randrange(0,len(dataset))
x = dataset[idx].to(device).unsqueeze(dim=0)

# generate several reconstructions and average them

num_iter = 50
recon = torch.zeros((50,),device='cuda')

for _ in range(num_iter):
    v,_,_ = rbm(x)
    v = torch.squeeze(v).transpose(-2,-1).argmax(dim=1)
    recon += v

final_v = recon / num_iter
final_x = torch.squeeze(x).transpose(-2,-1).argmax(dim=1)
