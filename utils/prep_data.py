import pandas as pd
import numpy as np

# load data from csv file

data_path = '../../../data/projG/data.csv'
headers = ['movie_num','user_num2','user_rating']
data = pd.read_csv(data_path,names = headers)

# add a user number column

num_movies = 50
users = np.arange(1,501,1,dtype = np.int)
users = np.tile(users,num_movies).reshape(num_movies*len(users),1)
data.insert(1,'user_num',users)

# delete the old user number column

data.drop('user_num2',axis = 1,inplace=True)

# check if there are any NaN values

# where_NaN = data.index[np.isnan(data).any(1)]

# sort the data by user number

data.sort_values(by = 'users',
                 inplace = True)

# write to csv file

data.to_csv('../../../data/projG/data_processed.csv',
            header = False,
            index = False)