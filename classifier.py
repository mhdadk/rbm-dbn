import torch
from sklearn.linear_model import SGDClassifier
import os
import time
import joblib

from CSVDataset_clf import CSVDataset
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
train_path = os.path.join(data_dir,'data_train.csv')
val_path = os.path.join(data_dir,'data_val.csv')
test_path = os.path.join(data_dir,'data_test.csv')
dataloaders = {}

# batch sizes for training, validation, and testing

train_batch_size = 64
val_batch_size = 64
test_batch_size = 64

for mode,path,batch_size in [('train',train_path,train_batch_size),
                             ('val',val_path,val_batch_size),
                             ('test',test_path,test_batch_size)]:
    
    dataset = CSVDataset(path)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True)

# initialize RBM and DBN classifiers

RBM_clf = SGDClassifier(loss = 'hinge',
                        penalty = 'l2',
                        alpha = 0.0001,
                        fit_intercept = True,
                        tol = 0.001,
                        shuffle = True,
                        n_jobs = -1,
                        random_state = 42,
                        learning_rate='optimal',
                        eta0 = 0.0,
                        power_t = 0.5,
                        early_stopping = False,
                        n_iter_no_change = 5,
                        class_weight = None,
                        warm_start = False,
                        average = False)

DBN_clf = SGDClassifier(loss = 'hinge',
                        penalty = 'l2',
                        alpha = 0.0001,
                        fit_intercept = True,
                        tol = 0.001,
                        shuffle = True,
                        n_jobs = -1,
                        random_state = 42,
                        learning_rate='optimal',
                        eta0 = 0.0,
                        power_t = 0.5,
                        early_stopping = False,
                        n_iter_no_change = 5,
                        class_weight = None,
                        warm_start = False,
                        average = False)

# number of epochs to train the classifier for

num_epochs = 10

# to know when to save parameters

best_RBM_val_acc = 0
best_DBN_val_acc = 0

if __name__ == '__main__':
    
    # starting time

    start = time.time()
    
    for epoch in range(num_epochs):
        
        # show number of epochs elapsed
        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        
        # record the epoch start time
        
        epoch_start = time.time()
        
        # training #######################################################
        
        # to compute average training accuracy over epoch
        
        RBM_train_acc = 0
        DBN_train_acc = 0
        
        print('Training...')
        
        for i,(y,x) in enumerate(dataloaders['train']):
            
            # convert to numpy
            
            y = y.numpy()
            
            # track progress
            
            print('\rProgress: {:.2f}%'.format(i*dataloaders['train'].batch_size/
                                             len(dataloaders['train'].dataset)*100),
                  end='',flush=True)
            
            # get batch of hidden vectors from RBM and DBN. Also, remove
            # the last dimension to be used later in partial_fit
            
            h_RBM = rbm(x)[1].squeeze().cpu().numpy()
            h_DBN = dbn(x)[1].squeeze().cpu().numpy()
            
            # train classifiers over batch
            
            RBM_clf.partial_fit(h_RBM,y,[1,2,3,4,5])
            DBN_clf.partial_fit(h_DBN,y,[1,2,3,4,5])
            
            # get training accuracy
            
            RBM_train_acc += RBM_clf.score(h_RBM,y)
            DBN_train_acc += DBN_clf.score(h_DBN,y)
            
        average_RBM_train_acc = RBM_train_acc / len(dataloaders['train'].dataset)
        average_DBN_train_acc = DBN_train_acc / len(dataloaders['train'].dataset)
        
        # show results
            
        print('\nRBM Training Accuracy: {:.2f}%'.format(average_RBM_train_acc*100))
        print('\nDBN Training Accuracy: {:.2f}%'.format(average_DBN_train_acc*100))
        
        # validation ######################################################
        
        # to compute average validation accuracy over epoch
        
        RBM_val_acc = 0
        DBN_val_acc = 0
        
        print('Validating...')
        
        for i,(y,x) in enumerate(dataloaders['val']):
            
            # get class numbers for partial_fit
            
            classes = torch.unique(y).numpy()
            
            # convert to numpy
            
            y = y.numpy()
            
            # track progress
            
            print('\rProgress: {:.2f}%'.format(i*dataloaders['val'].batch_size/
                                             len(dataloaders['val'].dataset)*100),
                  end='',flush=True)
            
            # get batch of hidden vectors from RBM and DBN. Also, remove
            # the last dimension to be used later in partial_fit
            
            h_RBM = rbm(x)[1].squeeze().cpu().numpy()
            h_DBN = dbn(x)[1].squeeze().cpu().numpy()
                        
            # get validation accuracy
            
            RBM_val_acc += RBM_clf.score(h_RBM,y)
            DBN_val_acc += DBN_clf.score(h_DBN,y)
            
        average_RBM_val_acc = RBM_val_acc / len(dataloaders['val'].dataset)
        average_DBN_val_acc = DBN_val_acc / len(dataloaders['val'].dataset)
        
        # show results
            
        print('\nRBM Validation Accuracy: {:.2f}%'.format(average_RBM_val_acc*100))
        print('\nDBN Validation Accuracy: {:.2f}%'.format(average_DBN_val_acc*100))
        
        # show epoch time
            
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
        print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
    
        # save the parameters for the best validation accuracy for RBM
        
        if average_RBM_val_acc > best_RBM_val_acc:
            print('Saving RBM checkpoint...')
            best_RBM_val_acc = average_RBM_val_acc
            joblib.dump(RBM_clf,'parameters/RBM_clf.joblib')
        
        # save the parameters for the best validation accuracy for DBN
        
        if average_DBN_val_acc > best_DBN_val_acc:
            print('Saving DBN checkpoint...')
            best_DBN_val_acc = average_DBN_val_acc
            joblib.dump(RBM_clf,'parameters/DBN_clf.joblib')
                
    # show training and validation time and best validation accuracies
    
    end = time.time()
    total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
    print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
    print('Best RBM Validation Accuracy: {:.2f}%'.format(best_RBM_val_acc*100))
    print('Best DBN Validation Accuracy: {:.2f}%'.format(best_DBN_val_acc*100))
