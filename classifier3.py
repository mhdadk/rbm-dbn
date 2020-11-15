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

num_RBMs = 1

dbn = DBN(num_RBMs = num_RBMs,
          num_visible = [50],
          num_hidden = [50],
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
rbm.load_state_dict(torch.load('RBM1_best_param.pt'))

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
    
    # given the visible units, compute p(h|v) and then p(v|p(h|v)) for the
    # RBM
    
    p_h_given_v = rbm.compute_p_h_given_v(x)
    p_v_given_h = rbm.compute_p_v_given_h(p_h_given_v)
    
    # compute the mode of each probability distribution and compare to
    # x using balanced accuracy
    
    RBM_test_acc = 0
    
    for i,batch in enumerate(p_v_given_h):
        y_true = x[i].argmax(dim=0).squeeze().numpy()
        y_pred = batch.argmax(dim=-1).numpy()
        RBM_test_acc += balanced_accuracy_score(y_true,y_pred)
    
    # average over entire batch
    
    RBM_test_acc /= len(p_v_given_h)  
    
    # propagate probabilities forward in DBN
    
    v = x
    
    for RBM in dbn.rbms:
        v = RBM.compute_p_h_given_v(v)
    
    # propagate probabilities backward in DBN
    
    for RBM in reversed(dbn.rbms):
        v = RBM.compute_p_v_given_h(v)
    
    # compute balanced accuracy for DBN
    
    DBN_test_acc = 0
    
    for i,batch in enumerate(v):
        y_true = x[i].argmax(dim=0).squeeze().numpy()
        y_pred = batch.argmax(dim=-1).numpy()
        DBN_test_acc += balanced_accuracy_score(y_true,y_pred)
    
    # average over entire batch
    
    DBN_test_acc /= len(v)
    