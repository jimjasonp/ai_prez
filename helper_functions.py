import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
from scipy import signal
import pywt

def rfecv(X_train,y,X_test):
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeRegressor
    rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select = 3 )
    rfe.fit(X_train,y)
    for i,col in zip(range(X_train.shape[1]), X_train.columns):
        print(f"{col} selected = {rfe.support_[i]} rank = {rfe.ranking_[i]}")
    X_train = pd.DataFrame({'feature1':X_train[4],'feature2':X_train[10],'feature3':X_train[1]})
    rfe.transform(X_test)
    X_test = pd.DataFrame({'feature1':X_test[4],'feature2':X_test[10],'feature3':X_test[1]})

def pca(X_train,X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.7, random_state = 42)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pca.transform(X_test)
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    #plt.show()
    return X_train,X_test

def res_plot(model_list,min,mid,max,name_list):

    X_axis = np.arange(len(model_list)) 

    plt.bar(X_axis - 0.25, min, 0.2, label = '75 training samples')
    
    #for index, value in enumerate(min):
    #    plt.text(value, index,str(value))
    
    plt.bar(X_axis , mid, 0.2, label = '112 training samples')
    #for index, value in enumerate(mid):
    #    plt.text(value, index,str(value))
    
    plt.bar(X_axis + 0.25 , max, 0.2, label = '225 training samples')
    #for index, value in enumerate(max):
    #    plt.text(value, index,str(value))
    
    plt.xticks(X_axis, name_list)
    plt.xlabel("Models")
    plt.ylabel("Mean absolute Percentage error")
    plt.title(f"MAPE of models with different training sizes ")
    plt.legend() 
    plt.show()

def model_run(model,X_train,y,X_test):

    y_pred = model(X_train,y,X_test)
    #print(y_pred)
    y_true = [0.02,0.034,0.062,0.086,0.12]
    #y_true = y_set('Damage_percentage',r'C:\Users\jimja\Desktop\thesis\dokimes','regression')
    mape = 100*mean_absolute_percentage_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    return mae,mape,y_true,y_pred

def fourier(sample_sensor):
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    return power_spectrum,freqs


def pwelch(sample_sensor):
    fs = 1000
    (f, S)= signal.welch(sample_sensor, fs, nperseg=1024)
    return S
    #plt.semilogy(f, S)
    #plt.xlim([0, 500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def psd(sample_sensor):
    fs = 1000
    # f contains the frequency components
    # S is the PSD
    (f, S) = signal.periodogram(sample_sensor, fs, scaling='density')
    return S
    #plt.semilogy(f, S)
    #plt.ylim([1e-14, 1e-3])
    #plt.xlim([0,500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def spectrogram(sample):
    fs = 1000
    f, t, Sxx = signal.spectrogram(sample, fs)
    #plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    return Sxx

def wavelet(sample):
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
    signal = sample

    # Perform wavelet transform
    wavelet_name = 'db1' # Daubechies wavelet, order 1
    transformed_signal, _ = pywt.dwt(signal, wavelet_name)
    return transformed_signal
    # Plot the original signal
    #plt.subplot(2, 1, 1)
    #plt.plot(signal)
    #plt.title('Original Signal')

    # Plot the transformed signal
    #plt.subplot(2, 1, 2)
    #plt.plot(transformed_signal)
    #plt.title('Transformed Signal')

    #plt.tight_layout()
    #plt.show()

def signal_data(sample):
    from scipy import signal
    high_peaks, high_peaks_properties = signal.find_peaks(sample,prominence=0.08)
    low_peaks, low_peaks_properties = signal.find_peaks(sample,distance = 500,height=(0.005,0.008))
    dx = low_peaks - high_peaks
    dy = high_peaks_properties['prominences'] - low_peaks_properties['peak_heights']
    signal_props = [high_peaks_properties['prominences'],
                    high_peaks,
                    low_peaks_properties['peak_heights'],
                    low_peaks,
                    dx,
                    dy
                    ]
    return signal_props