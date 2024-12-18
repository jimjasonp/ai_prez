import numpy as np
from test_file_opener import y_set,X_set
from sklearn.svm import SVR,SVC
from sklearn.gaussian_process.kernels import ExpSineSquared,Product,RBF
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


############## Regression##############

# transformations fourier,none,wavelet,psd,pwelch
#transformation = 'none'

#X_data = X_set(r'C:\Users\jimja\Desktop\thesis\data',transformation)
#y_data = y_set(r'C:\Users\jimja\Desktop\thesis\data')


#X_random_data = X_set(r'C:\Users\jimja\Desktop\thesis\random_data',transformation)
#y_random_data = y_set(r'C:\Users\jimja\Desktop\thesis\random_data')

#X_dokimes = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes',transformation)

#X_train = np.concatenate((X_data,X_random_data),axis=0)
#y = np.concatenate((y_data,y_random_data),axis=0)
#X_test = X_dokimes

##########################################



############## classification##############

from x_y_set_dm_df_dd import X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


########################################################



sin_kernel = ExpSineSquared(length_scale=1, 
                            periodicity=1, 
                            length_scale_bounds=(10, 100.0), 
                            periodicity_bounds=(10, 100.0)
                            )

rbf = RBF(length_scale=100, 
          length_scale_bounds=(100, 100.0))


custom_kernel = Product(sin_kernel,rbf)

svm = SVC(
    C = 0.1,
    tol =1e-5 ,
    shrinking= True,
    max_iter=-1,
    kernel=sin_kernel
    )
svm.fit(X_train, y_train)




gp = GaussianProcessClassifier(
    kernel=custom_kernel,  
    optimizer='fmin_l_bfgs_b', 
    n_restarts_optimizer=0, 
)
gp.fit(X_train,y_train)


y_pred = svm.predict(X_test)
#print(y_pred)


CM = confusion_matrix(y_test,y_pred)
print(CM)
accuracy = accuracy_score(y_test, y_pred)
print('===============')
print("Accuracy:", accuracy)