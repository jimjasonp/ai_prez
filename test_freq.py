from test_file_opener import y_set,X_set
from models import *
from helper_functions import res_plot,model_run
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# transformations fourier,none,wavelet,psd,pwelch
transformation = 'fourier'

X_data = X_set(r'C:\Users\jimja\Desktop\thesis\data',transformation)
y_data = y_set(r'C:\Users\jimja\Desktop\thesis\data')


X_random_data = X_set(r'C:\Users\jimja\Desktop\thesis\random_data',transformation)
y_random_data = y_set(r'C:\Users\jimja\Desktop\thesis\random_data')

X_dokimes = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes',transformation)


#scaler = StandardScaler()
scaler = MinMaxScaler()
X_random_data= scaler.fit_transform(X_random_data)
X_data = scaler.transform(X_data)
X_dokimes = scaler.transform(X_dokimes)
X_dokimes = pd.DataFrame(X_dokimes)


plt.plot(X_random_data)
#plt.yscale('log')
plt.xlim(0,350)
plt.grid(True)
plt.show()