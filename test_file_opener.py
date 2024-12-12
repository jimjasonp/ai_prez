
import pandas as pd
import glob
import os
import numpy as np
import statistics
from helper_functions import pwelch,psd,fourier,wavelet,spectrogram,signal_data

#path = r'C:\Users\jimja\Desktop\thesis\data'
#path = r'C:\Users\jimja\Desktop\thesis\data' # use your path
# to sensor data list einai auto pou einai sth morfh gia train



def X_set(path,transformation):
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

    signal_properties_list =[]
    power_spectrum_list = []
    sensor_names = ['s2','s3','s4']
    for sensor in sensor_names:
        #gia kathe sample sensora dld gia kathe xronoseira (pou prokuptei apo to shma pou lambanei o sensoras efarmozo transformations
        for i in range(0,len(sensor_data_list)):
            sample_sensor =sensor_data_list[i][sensor]
            if transformation == 'fourier':
                power_spectrum = fourier(sample_sensor)
                #signal_props = signal_data(power_spectrum)
            elif transformation == 'psd':
                power_spectrum = psd(sample_sensor)
                #signal_props = signal_data(power_spectrum)
            elif transformation == 'pwelch':
                power_spectrum = pwelch(sample_sensor)
                #signal_props = signal_data(power_spectrum)
            elif transformation == 'wavelet':
                power_spectrum = wavelet(sample_sensor)
                #signal_props = signal_data(power_spectrum)
            elif transformation == 'none':
                power_spectrum = sample_sensor
                #signal_props = signal_data(power_spectrum)
            elif transformation == 'spectrogram':
                power_spectrum = spectrogram(sample_sensor)
                #signal_props = signal_data(power_spectrum)
            power_spectrum_list.append(power_spectrum)  
            #signal_properties_list.append(signal_props)

    sensor2_vector = []
    sensor3_vector = []
    sensor4_vector = []


    sensor2_signal_props =[]
    sensor3_signal_props =[]
    sensor4_signal_props =[]

    bound_1 = int(len(power_spectrum_list)/3)
    bound_2 = int(2*len(power_spectrum_list)/3)
    bound_3 = int(len(power_spectrum_list))


    for i in range(0,bound_1):
        sensor2_vector.append(power_spectrum_list[i])
        #sensor2_signal_props.append(signal_properties_list[i])

    for i in range(bound_1,bound_2):
        sensor3_vector.append(power_spectrum_list[i])
        #sensor3_signal_props.append(signal_properties_list[i])

    for i in range(bound_2,bound_3):
        sensor4_vector.append(power_spectrum_list[i])
        #sensor4_signal_props.append(signal_properties_list[i])

    X = np.concatenate((sensor2_vector,sensor3_vector,sensor4_vector),axis=1)
    #sig = np.concatenate((sensor2_signal_props,sensor3_signal_props,sensor4_signal_props),axis=1)
    return X#,sig


def y_set(path):
    dmg_list=[]
    name_list=[]
    damage_result = str('Damage_percentage')
    # gia kathe file name sto path pou exw dwsei afairei to .csv kai afairei nan values kai kanei mia lista mono me to damage percentage
    for filename in sorted(glob.glob(os.path.join(path , "meta*"))):
        df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
        dmg_perc = df[f'{damage_result}']
        if len(dmg_perc)== 1:
            dmg_perc = dmg_perc[0]
        dmg_list.append(dmg_perc)
        filename = filename.removesuffix('.csv')
        name_list.append(filename)

    # ftiaxnei ena dataframe me to damage percentage kai prosthetei to index number kai kanei sort basei autou 
    dmg_data = pd.DataFrame({'dmg':dmg_list,'damage_file_name':name_list})
    dmg_data['dmg_index_number'] = [int(i.split('_')[-1]) for i in dmg_data['damage_file_name']]
    dmg_data = dmg_data.sort_values(by=['dmg_index_number'])
    dmg_data = dmg_data.drop(['damage_file_name'],axis=1)
    damage_instances = dmg_data['dmg']
    damage_data_df = pd.DataFrame({f'{damage_result}':damage_instances})
    return damage_data_df
