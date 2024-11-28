import pandas as pd
import glob
import os
import math

path = r'C:\Users\jimja\Desktop\thesis\data' # use your path
dmg_list = []
name_list = []


damage_results_list = ['Damage_percentage', 'DamageLayer1', 'DamageLayer2', 'DamageLayer3', 'DamageLayer4', 'DamageLayer5']


def y_set_creator(damage_result,mode):

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








path = r'C:\Users\jimja\Desktop\thesis\data' # use your path
dmg_list = []
name_list = []

# gia kathe file name sto path pou exw dwsei afairei to .csv kai afairei nan values kai kanei mia lista mono me to damage percentage
for filename in sorted(glob.glob(os.path.join(path , "meta*"))):
    df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
    dmg_perc = df['Damage_percentage']
    if len(dmg_perc)== 1:
        dmg_perc = dmg_perc[0]
    dmg_list.append(dmg_perc)
    filename = filename.removesuffix('.csv')
    name_list.append(filename)





# ftiaxnei ena dataframe me to damage percentage kai prosthetei to index number kai kanei sort basei autou 
dmg_data = pd.DataFrame({'dmg':dmg_list,'damage_file_name':name_list})
dmg_data['dmg_index_number'] = [int(i.split('_')[-1]) for i in dmg_data['damage_file_name']]
dmg_data = dmg_data.sort_values(by=['dmg_index_number'])

import pandas as pd
from y_set_creator_dmg_percentage import dmg_data
import math
DL1 = pd.DataFrame()
DL2 = pd.DataFrame()
DL3 = pd.DataFrame()
DL4 = pd.DataFrame()
DL5 = pd.DataFrame()


DL1_total = pd.DataFrame()
DL3_total = pd.DataFrame()
DL5_total = pd.DataFrame()

mode = 'regression'  #'classification','regression'

odd_layer =['DamageLayer1', 'DamageLayer3', 'DamageLayer5'] ###['DamageLayer1', 'DamageLayer2', 'DamageLayer3', 'DamageLayer4', 'DamageLayer5']

for layer in odd_layer:
    df_list = []
    dm_list = []
    dtotal_list = []
    i = 0
    for path in dmg_data['damage_file_name']:
        path = path + '.csv'
        dataframe = pd.read_csv(path,sep=' |,', engine='python')
        DL = dataframe[layer]
        df = 1 - (1-DL[0])*(1-DL[1])
        dm = 1 - (1-DL[2])*(1-DL[3])
        if dm ==0:
            dm_list.append('undamaged')
        else :
            dm_list.append('damaged')
        if df ==0:
            df_list.append('undamaged')
        else :
            df_list.append('damaged')


        if df and dm ==0:
            dtotal_list.append('undamaged')
            i = i+1
        else :
            dtotal_list.append('damaged')

    if layer == 'DamageLayer1':
        DL1_total = pd.DataFrame({'dtotal': dtotal_list})
    if layer == 'DamageLayer3':
        DL3_total = pd.DataFrame({'dtotal': dtotal_list})
    if layer == 'DamageLayer5':
        DL5_total = pd.DataFrame({'dtotal': dtotal_list})

    if layer == 'DamageLayer1':
        DL1 = pd.DataFrame({'df': df_list,'dm':dm_list})
    if layer == 'DamageLayer3':
        DL3 = pd.DataFrame({'df': df_list,'dm':dm_list})
    if layer == 'DamageLayer5':
        DL5 = pd.DataFrame({'df': df_list,'dm':dm_list})


even_layer = [ 'DamageLayer2',  'DamageLayer4']

for layer in even_layer:
    dd_list = []
    for path in dmg_data['damage_file_name']:
        path = path + '.csv'
        dataframe = pd.read_csv(path,sep=' |,', engine='python')
        DL = dataframe[layer]
        dd = 1 - DL[0]
        dd = math.floor(dd*10)/10

        if dd ==0:
            dd_list.append('undamaged')
        else :
            dd_list.append('damaged')

    if layer == 'DamageLayer2':
        DL2 = pd.DataFrame({'dd': dd_list})
    if layer == 'DamageLayer4':
        DL4 = pd.DataFrame({'dd': dd_list})

