
import pandas as pd
import glob
import os
import numpy as np
import statistics
path = r'C:\Users\jimja\Desktop\thesis\data'
#path = r'C:\Users\jimja\Desktop\thesis\data' # use your path
# to sensor data list einai auto pou einai sth morfh gia train

def fourier(sample_sensor):
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    return power_spectrum

def X_set(path):
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



#gia kathe feature kataskeuazo ena dataframe pou tha mpoun gia kathe sensora oi times tou feature gia kathe timeserie tou sensora

    power_spectrum_list = []
    power_spectrum_fft_list = []
    # h diadikasia ginetai epanalhptika gia kathe feature sto feature list
    sensor_names = ['s2','s3','s4']
    for sensor in sensor_names:
        #gia kathe sample sensora dld gia kathe xronoseira (pou prokuptei apo to shma pou lambanei o sensoras efarmozo fft
        for i in range(0,len(sensor_data_list)):
            #efarmozo to metasxhmatismo fourier (fft) se kathe timeserie
            sample_sensor =sensor_data_list[i][sensor]
            power_spectrum = sample_sensor
            power_spectrum_list.append(power_spectrum)
            power_spectrum_fft = fourier(sample_sensor)
            power_spectrum_fft_list.append(power_spectrum_fft)  


    sensor2_vector = []
    sensor3_vector = []
    sensor4_vector = []

    bound_1 = int(len(power_spectrum_list)/3)
    bound_2 = int(2*len(power_spectrum_list)/3)
    bound_3 = int(len(power_spectrum_list))


    for i in range(0,bound_1):
        sensor2_vector.append(power_spectrum_list[i])

    for i in range(bound_1,bound_2):
        sensor3_vector.append(power_spectrum_list[i])

    for i in range(bound_2,bound_3):
        sensor4_vector.append(power_spectrum_list[i])

    sensor2_fft_vector = []
    sensor3_fft_vector = []
    sensor4_fft_vector = []

    for i in range(0,bound_1):
        sensor2_fft_vector.append(power_spectrum_fft_list[i])

    for i in range(bound_1,bound_2):
        sensor3_fft_vector.append(power_spectrum_fft_list[i])

    for i in range(bound_2,bound_3):
        sensor4_fft_vector.append(power_spectrum_fft_list[i])


    X_time = np.concatenate((sensor2_vector,sensor3_vector,sensor4_vector),axis=1)
    X_fft = np.concatenate((sensor2_fft_vector,sensor3_fft_vector,sensor4_fft_vector),axis=1)
    return X_time,X_fft





def y_set(damage_result,path,mode):
    dmg_list=[]
    name_list=[]
    damage_result = str(damage_result)
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





###### apo edw kai panw einai idio me y_set_for_layer
    #krataw ksexwrista ta index numbers wste na to perasw sto sensors gia na dw poia indeces den uparxoun
    dmg_index_list = dmg_data['dmg_index_number']

    dmg_data = dmg_data.drop(['damage_file_name'],axis=1)
    damage_instances = dmg_data['dmg']
    new_dmg = []

    # meta ftiaxnei mia nea lista me to damage percentage sth sosth seira
    for dmg in damage_instances:
        new_dmg.append(dmg)

    new_damage_data = pd.DataFrame({'damage_perc':new_dmg,'damage_index_number':dmg_data['dmg_index_number']})

    # to damage data df einai to damage sth morfh gia train epeidh kanw classification thelo na exw labels gia auto kanw to damage percentage string
    damage_data_list=[]
    for dmg in damage_instances:
        if mode =='classification':
            damage_data_list.append(str(dmg))
        if mode =='regression':
            damage_data_list.append(dmg)
    damage_data_df = pd.DataFrame({f'{damage_result}':damage_data_list})
    return damage_data_df


#print(X_set(r'C:\Users\jimja\Desktop\thesis\data')[0])
#print(y_set('Damage_percentage',r'C:\Users\jimja\Desktop\thesis\data','regression'))