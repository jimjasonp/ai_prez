from test_file_opener import y_set,X_set
from models import *
from helper_functions import res_plot,model_run
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
# transformations fourier,none,wavelet,psd,pwelch
import numpy as np
transformation = 'none'

def fourier(sample_sensor):
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    return power_spectrum,freqs



data = X_set(r'C:\Users\jimja\Desktop\thesis\random_data',transformation)
sample = data[20]

def fourier_signal_std(sample):
    amp= (fourier(sample)[0])
    freq= (fourier(sample)[1])

    amp_list =[]
    freq_list =[]
    bound = int(0.5*len(amp))
    max_freq = freq[amp.argmax()]

    for i in range(0,bound):
        amp_list.append(amp[i])
        freq_list.append(freq[i]/max_freq)

    amp = amp_list
    freq = freq_list
    return freq,amp

freq,amp = fourier_signal_std(sample)


plt.scatter(freq,amp)
plt.yscale('log')
#plt.xlim(0,1024)
plt.grid(True)
plt.show()

