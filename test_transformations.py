import numpy as np
from test_file_opener import y_set,X_set
from scipy import signal
import matplotlib.pyplot as plt

X_train = np.concatenate((
                        X_set(r'C:\Users\jimja\Desktop\thesis\random_data','none'),
                        X_set(r'C:\Users\jimja\Desktop\thesis\data','none')),
                        axis=0)


y = np.concatenate((
                        y_set(r'C:\Users\jimja\Desktop\thesis\random_data'),
                        y_set(r'C:\Users\jimja\Desktop\thesis\data')),
                        axis=0)
X_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes','none')
sample = X_train[0]

def pwelch(sample):
    fs = 1000
    (f, S)= signal.welch(sample, fs, nperseg=512)

    plt.semilogy(f, S)
    plt.xlim([0, 500])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

def psd(sample):
    fs = 1000
    # f contains the frequency components
    # S is the PSD
    (f, S) = signal.periodogram(sample, fs, scaling='density')

    plt.semilogy(f, S)
    plt.ylim([1e-14, 1e-3])
    plt.xlim([0,500])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

psd(sample)