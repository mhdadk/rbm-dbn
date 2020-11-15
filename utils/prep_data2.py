import pandas as pd
import numpy as np

# load data from csv file

data_path = '../../../data/projG/data.csv'
headers = ['movie_num','user_num2','rating']
data = pd.read_csv(data_path,names = headers)

# add a user number column

num_movies = 50
users = np.arange(1,501,1,dtype = np.int)
users = np.tile(users,num_movies).reshape(num_movies*len(users),1)
data.insert(1,'user_num',users)

# delete the old user number column

data.drop('user_num2',axis = 1,inplace=True)

#%%

num_samples = 5000000
num_new_data_per_movie = int(num_samples/num_movies) - 500
new_user_nums = np.arange(501,501 + num_new_data_per_movie,1,
                          dtype = np.int).reshape((num_new_data_per_movie,1))

for i in range(50):
    new_movie_nums = np.ones((num_new_data_per_movie,1),dtype=np.int) * (i + 1)
    
    new_ratings = np.random.randint(1,6,
                                    size = num_new_data_per_movie).reshape((num_new_data_per_movie,1))
    
    new_rand_data = np.concatenate((new_movie_nums,
                                    new_user_nums,
                                    new_ratings),axis = 1)
    
    new_rand_data = pd.DataFrame(data = new_rand_data,
                                 columns = ['movie_num','user_num','rating'])
    
    # insert new data
    
    if i < 2:
        start = 500 + num_new_data_per_movie*i
    else:
        start = 500*i + num_new_data_per_movie*i
    
    data = pd.concat((data.iloc[:start],
                      new_rand_data,
                      data.iloc[start:]),
                      axis = 0)

#%%

# sort the data by user number

data.sort_values(by = 'user_num',
                 ascending = True,
                 inplace = True)

# rearrange the columns

data = data.reindex(columns = ['user_num','movie_num','rating'])

# write to csv file

data.to_csv('../../../data/projG/data_processed.csv',
            header = False,
            index = False)