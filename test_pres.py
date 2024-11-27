
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sensor_mean = pd.read_csv('sensor_mean')

sensor_mean = sensor_mean[['s2', 's3','s4']]

dmg_perc = pd.read_csv('dmg_perc')

dmg_perc = dmg_perc[['Damage_percentage']]

damage_list = []


print(len(dmg_perc['Damage_percentage']))


for i in range(0,len(dmg_perc['Damage_percentage'])):
    #damage_list.append(str(dmg_perc['Damage_percentage'][i]))
    damage_list.append(dmg_perc['Damage_percentage'][i])


from feature_vector import X,X_fft


X = X_fft #X,X_fft,sensor_mean
y = damage_list

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#########kanw test me tis nees times pou edwse o rekatsinas


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#lr.fit(X_fft,y)

from feature_vector_dokimes import X_dokimes,X_fft_dokimes
from y_dokimes import damage_data_df
import xgboost as xgb

xgb = xgb.XGBRegressor()
#xgb.fit(X,y)

from sklearn.svm import SVR
svr = SVR()
#svr.fit(X,y)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor() 
#dt.fit(X,y)


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.000000002,
              max_iter=100000,
              tol=1.496e-07,
              selection='random'
              )
#lasso.fit(X,y)

from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=0.000000005,
                l1_ratio=0.2,  
                max_iter=1000000,  
                #tol=0.0000001,  
                selection='cyclic'
                )
en.fit(X,y)




from sklearn.model_selection import cross_val_predict,LeaveOneOut,cross_val_score
X = X_fft_dokimes
y = damage_data_df
y_pred = en.predict(X)
y_true = [0.2,0.034,0.062,0.086,0.012]
print('mape is')
print(mean_absolute_percentage_error(y_true,y_pred))
print('mae is')
print(mean_absolute_error(y_true,y_pred))
print(y_pred)