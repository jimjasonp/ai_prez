import numpy as np
from test_file_opener import y_set,X_set
import matplotlib.pyplot as plt
from helper_functions import fourier,model_run
from sklearn.preprocessing import StandardScaler

from models import linear_regression



X_train = X_set(r'C:\Users\jimja\Desktop\thesis\random_data','fourier')
X_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes','fourier')




#raw_train = X_set(r'C:\Users\jimja\Desktop\thesis\random_data','none')
#raw_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes','none')

#X_train =[]
#for sample in raw_train:
#    X_train.append(fourier(sample))

#X_test=[]
#for sample in raw_test:
#    X_test.append(fourier(sample))



y_train = y_set(r'C:\Users\jimja\Desktop\thesis\random_data')

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

model = linear_regression

print(model_run(model,X_train,y_train,X_test)[1])

