import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from test_file_opener import y_set,X_set
from models import *
from helper_functions import res_plot,model_run

X_data = X_set(r'C:\Users\jimja\Desktop\thesis\data')[0]
y_data = y_set(r'C:\Users\jimja\Desktop\thesis\data')


X_random_data = X_set(r'C:\Users\jimja\Desktop\thesis\random_data')[0]
y_random_data = y_set(r'C:\Users\jimja\Desktop\thesis\random_data')

X_dokimes = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes')[0]

scaler = StandardScaler()
X_random_data= scaler.fit_transform(X_random_data)
X_data = scaler.transform(X_data)
X_dokimes = scaler.transform(X_dokimes)
X_dokimes = pd.DataFrame(X_dokimes)

model_list = [mlp,elastic_net,linear_regression,ridge_reg,random_forest_reg]
name_list = ['mlp','elastic net','linear regression','ridge regression','random forest']


max = [] # kai ta duo sets einai full
mid = [] # kai ta duo sets einai misa
min = [] # mono to random_data


#### max krataw olo to random dataset kai olo to original
X_train = np.concatenate((X_data,X_random_data),axis=0)
y = np.concatenate((y_data,y_random_data),axis=0)
X_test = X_dokimes
for model in model_list:
    max.append(model_run(model,X_train,y,X_test)[1]) 


#### mid krataw to miso random dataset kai to miso original

X_data_half, X_drop, y_data_half, y_drop = train_test_split(X_data, y_data, test_size=0.5,shuffle=True)
X_random_data_half, X_drop, y_random_data_half, y_drop = train_test_split(X_random_data, y_random_data, test_size=0.5,shuffle=True)
X_train_half = np.concatenate((X_data_half,X_random_data_half),axis=0)
y_half = np.concatenate((y_data_half,y_random_data_half),axis=0)
X_test = X_dokimes
for model in model_list:
    mid.append(model_run(model,X_train_half,y_half,X_test)[1]) 


#### min krataw mono to random

X_train = X_random_data
y = y_random_data
X_test = X_dokimes
for model in model_list:
    min.append(model_run(model,X_train,y,X_test)[1])



res_plot(model_list,min,mid,max,name_list)

