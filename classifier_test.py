import torch
import os
from sklearn.metrics import balanced_accuracy_score

from CSVDataset import CSVDataset
from models.DBN.DBN import DBN

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cpu')

# initialize DBN

num_RBMs = 3

dbn = DBN(num_RBMs = num_RBMs,
          num_visible = [50,25,15],
          num_hidden = [25,15,10],
          num_categories = 5,
          num_sampling_iter = [5,5,5],
          device = device)

# load RBM pretrained weights

param_dir = 'parameters'

for i in range(num_RBMs):
    param_path = os.path.join(param_dir,'RBM'+str(i)+'_param.pt')
    dbn.rbms[i].load_state_dict(torch.load(param_path))

# initialize RBM. This is the first RBM in the DBN

rbm = dbn.rbms[0]

# initialize dataloaders

data_dir = '../../data/projG'
test_path = os.path.join(data_dir,'data_test.csv')
dataset = CSVDataset(test_path)
batch_size = 64
dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = batch_size,
                                         shuffle = True)

x = next(iter(dataloader))

h_RBM = rbm(x)[1]
h_DBN = dbn(x)

# backpropagate probability in RBM

v_RBM = rbm.compute_p_v_given_h(h_RBM)

# backpropagate probabilities in DBN

v_DBN = h_DBN

for rbm in reversed(dbn.rbms):
    v_DBN = rbm.compute_p_v_given_h(v_DBN)

# compute accuracy of example for RBM

y_true = x[0].argmax(dim=0).squeeze().numpy()
y_pred = v_RBM[0].argmax(dim=-1).numpy()
RBM_score = balanced_accuracy_score(y_true,y_pred)

# compute accuracy of example for DBN

y_pred = v_DBN[0].argmax(dim=-1).numpy()
DBN_score = balanced_accuracy_score(y_true,y_pred)
