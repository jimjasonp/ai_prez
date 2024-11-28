
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



##### kanw train me to original dataset
from y_set_creator_dmg_percentage import y_set_creator
from feature_vector import X,X_fft


X = X_fft #X,X_fft,sensor_mean
y = y_set_creator('Damage_percentage','regression')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


############ kalw diafora montela############


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_fft,y)


import xgboost as xgb
xgb = xgb.XGBRegressor()
xgb.fit(X,y)

from sklearn.svm import SVR
svr = SVR()
svr.fit(X,y)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor() 
dt.fit(X,y)


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.000000002,
              max_iter=100000,
              tol=1.496e-07,
              selection='random'
              )
lasso.fit(X,y)

from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=0.000000005,
                l1_ratio=0.2,  
                max_iter=1000000,  
                #tol=0.0000001,  
                selection='cyclic'
                )
en.fit(X,y)

from sklearn.linear_model import Ridge

rr = Ridge(
        alpha=0.000000005,   
        max_iter=1000000, 
        tol=0.0000001, 
        solver='auto',  
        )
rr.fit(X,y)
################################################



################ PCA ################







################################################




##### kanw test me times rek


from feature_vector_dokimes_sample_rek import X_dokimes,X_fft_dokimes
from y_dokimes import damage_data_df

X = X_fft_dokimes

y_pred = rr.predict(X)
print(y_pred)

y_true = [0.02,0.034,0.062,0.086,0.12]


print('mape is')
print(mean_absolute_percentage_error(y_true,y_pred))
print('mae is')
print(mean_absolute_error(y_true,y_pred))
