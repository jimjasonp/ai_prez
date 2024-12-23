from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


##### kanw train me to original dataset
from y_set_creator_dmg_percentage import y_set_creator
#from feature_vector import X,X_fft


from feature_vector import sensor_mean,sensor_max,sensor_stdev,sensor_median_high,X,X_fft




'''
X_train = pd.DataFrame({
    's2_mean':sensor_mean['s2'],
    's3_mean':sensor_mean['s3'],
    's4_mean':sensor_mean['s4'],

    's2_max':sensor_max['s2'],
    's3_max':sensor_max['s3'],
    's4_max':sensor_max['s4'],

    's2_median_high':sensor_median_high['s2'],
    's3_median_high':sensor_median_high['s3'],
    's4_median_high':sensor_median_high['s4'],

    's2_stdev':sensor_stdev['s2'],
    's3_stdev':sensor_stdev['s3'],
    's4_stdev':sensor_stdev['s4'],
    })
'''
#X_train = X_fft #X,X_fft,sensor_mean
#y = y_set_creator('Damage_percentage','regression')



from test_file_opener import y_set,X_set

#X_train_1 = X_set(r'C:\Users\jimja\Desktop\thesis\data')[0]
#y_1 = y_set('Damage_percentage',r'C:\Users\jimja\Desktop\thesis\data','regression')

#size = 0.5


#X_train_1, X_drop, y_1, y_drop = train_test_split(X_train_1, y_1, test_size=size,shuffle=True)



#X_train_2 = X_set(r'C:\Users\jimja\Desktop\thesis\random_data')[0]
#y_2 = y_set('Damage_percentage',r'C:\Users\jimja\Desktop\thesis\random_data','regression')


#X_train = np.concatenate((X_train_1,X_train_2),axis=0)
#y = np.concatenate((y_1,y_2),axis=0)

X_train = X_set(r'C:\Users\jimja\Desktop\thesis\random_data')[0]
y = y_set('Damage_percentage',r'C:\Users\jimja\Desktop\thesis\random_data','regression')


print(len(X_train))




scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)

#X_train, X_drop, y, y_drop = train_test_split(X_train, y, test_size=0.01,shuffle=True)


#X_train = pd.DataFrame(X_train)

#################   RFECV   ##########################

#from sklearn.feature_selection import RFE
#from sklearn.tree import DecisionTreeRegressor
#rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select = 3 )
#rfe.fit(X_train,y)
#for i,col in zip(range(X_train.shape[1]), X_train.columns):
#    print(f"{col} selected = {rfe.support_[i]} rank = {rfe.ranking_[i]}")
#X_train = pd.DataFrame({'feature1':X_train[4],'feature2':X_train[10],'feature3':X_train[1]})
#rfe.transform(X_test)
#X_test = pd.DataFrame({'feature1':X_test[4],'feature2':X_test[10],'feature3':X_test[1]})
###############################################



#################     PCA     #####################
from sklearn.decomposition import PCA
pca = PCA(n_components=0.7, random_state = 42)
#pca.fit(X_train)
#X_train = pca.transform(X_train)
#X_train = pd.DataFrame(X_train)
#X_test = pca.transform(X_test)
#The following code constructs the Scree plot
#per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
#labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

#plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
#plt.ylabel('Percentage of Explained Variance')
#plt.xlabel('Principal Component')
#plt.title('Scree Plot')
#plt.show()
################################################






##### kanw test me times rek


from feature_vector_dokimes_sample_rek import X_dokimes,X_fft_dokimes,sensor_stdev,sensor_mean,sensor_max,sensor_median_high
from y_dokimes import damage_data_df
#X_test = X_dokimes
'''
X_test = pd.DataFrame({
's2_mean':sensor_mean['s2'],
's3_mean':sensor_mean['s3'],
's4_mean':sensor_mean['s4'],

's2_max':sensor_max['s2'],
's3_max':sensor_max['s3'],
's4_max':sensor_max['s4'],

's2_median_high':sensor_median_high['s2'],
's3_median_high':sensor_median_high['s3'],
's4_median_high':sensor_median_high['s4'],

's2_stdev':sensor_stdev['s2'],
's3_stdev':sensor_stdev['s3'],
's4_stdev':sensor_stdev['s4'],
})
'''
X_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes')[0]
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)






############ kalw diafora montela############
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,y)



from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.000000002,
        max_iter=100000,
        selection='random'
        )
#lasso.fit(X_train,y)

from sklearn.linear_model import ElasticNetCV

encv = ElasticNetCV(
            cv= 50,
            eps=1e-6,
            l1_ratio=0.01,  
            max_iter=100000,  
            tol=1e-6,  
            selection='cyclic'
            )
#encv.fit(X_train,y)

from sklearn.linear_model import ElasticNet

en = ElasticNet(
            #alpha = 0.000000002,
            l1_ratio=0.01,
            alpha = 0.2,  
            max_iter=100000000,  
            tol=0.000001,  
            selection='random'
            )
en.fit(X_train,y)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y)

from sklearn.linear_model import Ridge
rr = Ridge(
    alpha=1,   
    max_iter=1000000, 
    tol=0.0000001, 
    solver='auto',  
    )
rr.fit(X_train,y)


import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Flatten,Dense


mlp = Sequential()
# Flatten input from 28x28 images to 784 (28*28) vector
#mlp.add(Flatten(input_shape=(None, 2250)))

# Dense layer 1 (256 neurons)
mlp.add(Dense(256, activation='sigmoid'))

# Dense layer 2 (128 neurons)
mlp.add(Dense(128, activation='sigmoid'))


mlp.add(Dense(64, activation='sigmoid'))

#mlp.add(Dense(32, activation='sigmoid'))
# Output layer (10 classes)
mlp.add(Dense(10, activation='sigmoid'))

mlp.add(Dense(1, activation='linear'))


mlp.compile(loss="mean_absolute_error", optimizer="adam")

history = mlp.fit(X_train, y, epochs=150)

################################################


model_list = [ en, lr, rr]

def model_run(model):

    y_pred = model.predict(X_test)
    #print(y_pred)
    y_true = [0.02,0.034,0.062,0.086,0.12]
    #y_true = y_set('Damage_percentage',r'C:\Users\jimja\Desktop\thesis\dokimes','regression')
    mape = mean_absolute_percentage_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    return mae,mape,y_true,y_pred


# creating the dataset 

######### good models ################
data = {'Linear Regression':model_run(lr)[1], 
        'Ridge':model_run(rr)[1],
        'Elastic Net':model_run(en)[1],
        
        }

model_names  = list(data.keys())
mape = list(data.values())

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(model_names, mape, color ='maroon', 
        width = 0.4)

samples = len(X_train)
plt.xlabel("Models")
plt.ylabel("Mean absolute Percentage error")
plt.title(f"Mean absolute percentage error of models with number of samples used = {samples}")
plt.show()
################################################
data = {'MLP':model_run(mlp)[1], 
        'Decision Trees':model_run(dt)[1],
        'Random Forest':model_run(rf)[1]
        }

model_names  = list(data.keys())
mape = list(data.values())

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(model_names, mape, color ='maroon', 
        width = 0.4)

samples = len(X_train)
plt.xlabel("Models")
plt.ylabel("Mean absolute Percentage error")
plt.title(f"Mean absolute percentage error of models with number of samples used = {samples}")
plt.show()
################  bad models  ################


################################################
plt.plot(model_run(lr)[2],marker = 'o')
plt.plot(model_run(lr)[3],linestyle='dashed',marker = 'o')
plt.xlabel("sample")
plt.ylabel("y value")
plt.title(f" Predicted and true value of samples using Linear Regression")
plt.legend(["y_test", "y_pred"], loc="lower right")
#plt.show()