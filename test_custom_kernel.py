import numpy as np
from test_file_opener import y_set,X_set
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import ExpSineSquared,Product,RBF


X_train = np.concatenate((
                        X_set(r'C:\Users\jimja\Desktop\thesis\random_data')[0],
                        X_set(r'C:\Users\jimja\Desktop\thesis\data')[0]),
                        axis=0)


y = np.concatenate((
                        y_set(r'C:\Users\jimja\Desktop\thesis\random_data'),
                        y_set(r'C:\Users\jimja\Desktop\thesis\data')),
                        axis=0)
X_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes')[0]


sin_kernel = ExpSineSquared(length_scale=1.0, 
                            periodicity=1.0, 
                            length_scale_bounds=(1e-05, 100000.0), 
                            periodicity_bounds=(1e-05, 100000.0)
                            )

rbf = RBF(length_scale=1.0, 
          length_scale_bounds=(1e-05, 100000.0))


custom_kernel = Product(sin_kernel,rbf)

svr = SVR(
    C = 0.1,
    tol =1e-5 ,
    epsilon=0.1,
    shrinking= True,
    max_iter=-1,
    kernel=custom_kernel
    )
svr.fit(X_train, y)


from sklearn.gaussian_process import GaussianProcessRegressor

gpr = GaussianProcessRegressor(
    kernel=custom_kernel, 
    alpha=1e-10, 
    optimizer='fmin_l_bfgs_b', 
    n_restarts_optimizer=0, 
)


y_pred = gpr.predict(X_test)
print(y_pred)