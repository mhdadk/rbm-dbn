import torch
import os
import time
import sklearn

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
batch_size = 50
dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = batch_size,
                                         shuffle = False)

RBM_test_acc = 0
DBN_test_acc = 0

for i,x in enumerate(dataloader):
        
    # track progress
    
    print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                     len(dataloader.dataset)*100),
          end='',flush=True)
    
    # get batch of hidden vectors from RBM and DBN
    
    h_RBM = rbm(x)[1]
    h_DBN = dbn(x)
    
    # backpropagate probability p(v|h) in RBM
    
    v_RBM = rbm.compute_p_v_given_h(h_RBM)
    
    # compute balanced accuracy for RBM
    
    RBM_test_acc = 0
    
    for i,batch in enumerate(v_RBM):
        y_true = x[i].argmax(dim=0).squeeze().numpy()
        y_pred = batch.argmax(dim=-1).numpy()
        RBM_test_acc += sklearn.metrics.balanced_accuracy_score(y_true,y_pred)
    
    # average over entire batch
    
    RBM_test_acc /= len(v_RBM)  
    
    # backpropagate probability p(v|h) in DBN
    
    v_DBN = h_DBN
    
    for RBM in reversed(dbn.rbms):
        v_DBN = RBM.compute_p_v_given_h(v_DBN)
    
    # compute balanced accuracy for DBN
    
    DBN_test_acc = 0
    
    for i,batch in enumerate(v_DBN):
        y_true = x[i].argmax(dim=0).squeeze().numpy()
        y_pred = batch.argmax(dim=-1).numpy()
        DBN_test_acc += sklearn.metrics.balanced_accuracy_score(y_true,y_pred)
    
    # average over entire batch
    
    DBN_test_acc /= len(v_DBN)
    