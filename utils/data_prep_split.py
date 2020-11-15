import pandas as pd
from sklearn.model_selection import train_test_split
import os

data_path = '../../../data/projG/data.csv'

x = pd.read_csv(data_path,
                header = None)

x_train,x_val = train_test_split(x,
                                 train_size = 0.8,
                                 random_state = 42,
                                 shuffle = False)

x_val,x_test = train_test_split(x_val,
                                train_size = 0.5,
                                random_state = 42,
                                shuffle = False)

if not os.path.exists('../../../data/projG/data_train.csv'):
    
    x_train.to_csv('../../../data/projG/data_train.csv',
                   header = False,
                   index = False)

if not os.path.exists('../../../data/projG/data_val.csv'):
    
    x_val.to_csv('../../../data/projG/data_val.csv',
                 header = False,
                 index = False)

if not os.path.exists('../../../data/projG/data_test.csv'):
    
    x_test.to_csv('../../../data/projG/data_test.csv',
                  header = False,
                  index = False)
