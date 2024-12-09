from test_file_opener import y_set,X_set
from models import *
from helper_functions import res_plot,model_run
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
# transformations fourier,none,wavelet,psd,pwelch
transformation = 'fourier'



X_random_data = X_set(r'C:\Users\jimja\Desktop\thesis\random_data',transformation)

X_random_data = X_random_data[0]
freq = range(0,len(X_random_data))

df = pd.DataFrame({'magnitude':X_random_data,'freq':freq})
#scaler = StandardScaler()

scaler = MinMaxScaler()
df = scaler.fit_transform(df)
print(df)

plt.plot(df)
plt.yscale('log')
plt.xlim(0,1024)
plt.grid(True)
plt.show()