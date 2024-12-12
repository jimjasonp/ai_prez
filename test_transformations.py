import numpy as np
from test_file_opener import y_set,X_set
import matplotlib.pyplot as plt
from helper_functions import fourier
X_train = np.concatenate((
                        X_set(r'C:\Users\jimja\Desktop\thesis\random_data','none'),
                        X_set(r'C:\Users\jimja\Desktop\thesis\data','none')),
                        axis=0)


y = np.concatenate((
                        y_set(r'C:\Users\jimja\Desktop\thesis\random_data'),
                        y_set(r'C:\Users\jimja\Desktop\thesis\data')),
                        axis=0)
X_test = X_set(r'C:\Users\jimja\Desktop\thesis\dokimes','none')
sample = fourier(X_train[0])

#low_peaks, properties = signal.find_peaks(x,height=(0.005,0.008))

#plt.plot(x)
#plt.plot(peaks, x[peaks], "x")
#plt.plot(np.zeros_like(x), "--", color="gray")
#plt.show()

#print('high prominences')
#print(high_peaks_properties['prominences'][0])
#print('high freqs')
#print(high_peaks[0])

#print('low prominences')
#print(low_peaks_properties['peak_heights'][0])
#print('low freqs')
#print(low_peaks[0])
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



import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a sample signal
def wavelets(sample):
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

wavelets(sample)