import numpy as np
from test_file_opener import y_set,X_set
from scipy import signal
import matplotlib.pyplot as plt
import pywt
X_train = np.concatenate((
                        X_set(r'C:\Users\jimja\Desktop\thesis\random_data','none'),
                        X_set(r'C:\Users\jimja\Desktop\thesis\data','none')),
                        axis=0)


y = np.concatenate((
                        y_set(r'C:\Users\jimja\Desktop\thesis\random_data'),
                        y_set(r'C:\Users\jimja\Desktop\thesis\data')),
                        axis=0)
X_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes','none')


coeffs =  pywt.dwt2(X_train, 'bior1.3')
LL, (LH, HL, HH) = coeffs

print(len(y))