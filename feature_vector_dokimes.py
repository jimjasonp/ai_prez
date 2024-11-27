
import pandas as pd
import glob
import os
import numpy as np
import statistics
#path = r'C:\Users\jimja\Desktop\thesis\dokimes'
path = r'C:\Users\jimja\Desktop\thesis\dokimes' # use your path
# to sensor data list einai auto pou einai sth morfh gia train
sensor_data_list = []
name_list = []




# gia kathe filename sto path pou tou exw dwsei afairei to .csv wste meta na mporei na diabasei ton arithmo
for filename in sorted(glob.glob(os.path.join(path , "data*"))):
    filename = filename.removesuffix('.csv')
    name_list.append(filename)


#apo kathe filename krataei mono ton arithmo sto telos kai me auton ton arithmo ftiaxeni th nea sthlh index number
sensor_data = pd.DataFrame({'name':name_list})
sensor_data['sensor_index_number'] = [int(i.split('_')[-1]) for i in sensor_data['name']]


#kanw sort th lista basei tou index number
sensor_data = sensor_data.sort_values(by=['sensor_index_number'])


suffix='.csv'
new_names=[]


#se kathe filename sth lista pou exei ginei sort prosthetei to .csv wste na mporei na to diabasei
for filename in sensor_data['name']:
    filename = filename+suffix
    new_names.append(filename)



#anoigei ta arxeia apo kathe path kai ftiaxnei th lista me tis metrhseis

for filename in new_names:
    df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
    sensor_data_list.append(df)


feature_list = ['max','mean','stdev','median_high']

sensor_max = pd.DataFrame()
sensor_mean = pd.DataFrame()
sensor_stdev = pd.DataFrame()
sensor_median_high = pd.DataFrame()



def feature_maker(feature,sensor_fft,power_spectrum):
    if feature == 'max':
        sensor_fft.append(max(power_spectrum))
    elif feature =='mean':
        sensor_fft.append(statistics.mean(power_spectrum))
    elif feature =='stdev':
        sensor_fft.append(statistics.stdev(power_spectrum))
    elif feature =='median_high':
        sensor_fft.append(statistics.median_high(power_spectrum))
    return sensor_fft


def fourier(sample_sensor):
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    return power_spectrum

#gia kathe feature kataskeuazo ena dataframe pou tha mpoun gia kathe sensora oi times tou feature gia kathe timeserie tou sensora

power_spectrum_list = []
power_spectrum_fft_list = []
for feature in feature_list:
    # h diadikasia ginetai epanalhptika gia kathe feature sto feature list
    sensor_fft_df = pd.DataFrame()
    sensor_names = ['s2','s3','s4']
    for sensor in sensor_names:
        sensor_fft = []
        #gia kathe sample sensora dld gia kathe xronoseira (pou prokuptei apo to shma pou lambanei o sensoras efarmozo fft
        for i in range(0,len(sensor_data_list)):
            #efarmozo to metasxhmatismo fourier (fft) se kathe timeserie
            sample_sensor =sensor_data_list[i][sensor]

            power_spectrum = sample_sensor
            power_spectrum_list.append(power_spectrum)


            power_spectrum_fft = fourier(sample_sensor)
            power_spectrum_fft_list.append(power_spectrum_fft)

            # ta apotelesmata tou fft ta metatrepw se kapoio feature   
            sensor_fft = feature_maker(feature,sensor_fft,power_spectrum)
        # tis times tou kathe feature tis pernaw se ena df 
        new_data = {sensor: sensor_fft}
        sensor_fft_df = sensor_fft_df.assign(**new_data)
    #kataskeuazw ena dataframe gia to kathe feature me to antistoixo onoma
    if feature == 'max':
        sensor_max = sensor_max.assign(**sensor_fft_df)
    elif feature =='mean':
        sensor_mean = sensor_mean.assign(**sensor_fft_df)
    elif feature =='stdev':
        sensor_stdev = sensor_stdev.assign(**sensor_fft_df)
    elif feature =='median_high':
        sensor_median_high = sensor_median_high.assign(**sensor_fft_df)



sensor2_vector = []
sensor3_vector = []
sensor4_vector = []

for i in range(0,5):
    sensor2_vector.append(power_spectrum_list[i])

for i in range(5,10):
    sensor3_vector.append(power_spectrum_list[i])

for i in range(10,15):
    sensor4_vector.append(power_spectrum_list[i])

sensor2_fft_vector = []
sensor3_fft_vector = []
sensor4_fft_vector = []

for i in range(0,5):
    sensor2_fft_vector.append(power_spectrum_fft_list[i])

for i in range(5,10):
    sensor3_fft_vector.append(power_spectrum_fft_list[i])

for i in range(10,15):
    sensor4_fft_vector.append(power_spectrum_fft_list[i])


X_dokimes = np.concatenate((sensor2_vector,sensor3_vector,sensor4_vector),axis=1)
X_fft_dokimes = np.concatenate((sensor2_fft_vector,sensor3_fft_vector,sensor4_fft_vector),axis=1)

#print(len(X_fft_dokimes))
#print(len(X_dokimes))
#print(len(sensor_max))


print(X_dokimes)