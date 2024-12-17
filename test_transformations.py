import numpy as np
from test_file_opener import X_set,y_set
import matplotlib.pyplot as plt
from helper_functions import fourier
from sklearn.preprocessing import StandardScaler

set = X_set(r'C:\Users\jimja\Desktop\thesis\random_data','none')


X_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes','none')
################################################################
def signal_props_extract(sample):
    freq = sample[1]
    amp = sample[0]

    for i in range(0,len(freq)):
        if freq[i] == 200:
            first_bound = i
        if freq[i] == 400:
            second_bound = i
        if freq[i] ==0:
            zero_bound = i
    
    first_amp =[]
    for i in range(zero_bound,first_bound):
        first_amp.append(amp[i])

    second_amp =[]
    for i in range(first_bound,second_bound):
        second_amp.append(amp[i])
    
    for i in range(zero_bound,first_bound):
        if amp[i] == max(first_amp):
            first_max_amp = amp[i]
            first_max_freq = freq[i]

    for i in range(first_bound,second_bound):
        if amp[i] == max(second_amp):
            second_max_amp = amp[i]
            second_max_freq = freq[i]

    dx = second_max_freq-first_max_freq
    dy = first_max_amp-second_max_amp
    props = first_max_freq,first_max_amp,second_max_freq,second_max_amp,dx,dy
    
    return props
################################################################


feature_vector =[]
for sample in set:
    sample = fourier(sample)
    feature_vector.append(signal_props_extract(sample))

X_test_new=[]
for sample in X_test:
    sample = fourier(sample)
    X_test_new.append(signal_props_extract(sample))

X_test = X_test_new
X_train = feature_vector


def signal_data(sample):
    from scipy import signal
    high_peaks, high_peaks_properties = signal.find_peaks(sample,prominence=0.08)
    low_peaks, low_peaks_properties = signal.find_peaks(sample,distance = 500,height=(0.005,0.008))
    dx = low_peaks[0] - high_peaks[0]
    dy = high_peaks_properties['prominences'][0] - low_peaks_properties['peak_heights'][0]
    signal_props = [high_peaks_properties['prominences'][0],
                    high_peaks[0],
                    low_peaks_properties['peak_heights'][0],
                    low_peaks[0],
                    dx,
                    dy
                    ]
    return signal_props

y_train = y_set(r'C:\Users\jimja\Desktop\thesis\random_data')

from models import *

y_pred = linear_regression(X_train,y_train,X_test)

y_true = [0.02,0.034,0.062,0.086,0.12]
from sklearn.metrics import mean_absolute_percentage_error
mape=mean_absolute_percentage_error(y_pred,y_true)