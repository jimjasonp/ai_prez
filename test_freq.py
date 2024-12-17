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
sample = X_set(r'C:\Users\jimja\Desktop\thesis\random_data',transformation)
def fourier(sample_sensor):
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    return power_spectrum,freqs


def psd(sample_sensor):
    fs = 1000
    # f contains the frequency components
    # S is the PSD
    (f, S) = signal.periodogram(sample_sensor, fs, scaling='density')
    return S,f
    #plt.semilogy(f, S)
    #plt.ylim([1e-14, 1e-3])
    #plt.xlim([0,500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

amp = []
freq =[]
for point in sample:
    amp.append(fourier(point)[0])
    freq.append(fourier(point)[1])




amp = amp[20]
freq = freq[20]

amp_list =[]
freq_list =[]
bound = int(0.5*len(amp))



max_freq = freq[amp.argmax()]
#max_freq = 150
#max_freq-abs(freq[i]-max_freq
for i in range(0,bound):
    amp_list.append(amp[i])
    freq_list.append(freq[i]/max_freq)

amp = amp_list
freq = freq_list


amp = np.array(amp)
freq = np.array(freq)
freq = freq.reshape(-1, 1)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#freq= scaler.fit_transform(freq)


plt.scatter(freq,amp)
#plt.yscale('log')
#plt.xlim(0,1024)
plt.grid(True)
plt.show()

