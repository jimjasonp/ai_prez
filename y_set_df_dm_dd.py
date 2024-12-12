import pandas as pd
import math
import glob
import os
DL1 = pd.DataFrame()
DL2 = pd.DataFrame()
DL3 = pd.DataFrame()
DL4 = pd.DataFrame()
DL5 = pd.DataFrame()
layer_damage = pd.DataFrame()

path = r'C:\Users\jimja\Desktop\thesis\Balanced_Data'
dmg_list = []
name_list = []
case_list = []

damage_results_list = ['Damage_percentage', 'DamageLayer1', 'DamageLayer2', 'DamageLayer3', 'DamageLayer4', 'DamageLayer5']

damage_result = 'Damage_percentage'
# gia kathe file name sto path pou exw dwsei afairei to .csv kai afairei nan values kai kanei mia lista mono me to damage percentage
for filename in sorted(glob.glob(os.path.join(path , "meta*"))):
    df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
    dmg_perc = df[f'{damage_result}']
    case = df['caseStudey'][0]
    if len(dmg_perc)== 1:
         dmg_perc = dmg_perc[0]
    dmg_list.append(dmg_perc)
    filename = filename.removesuffix('.csv')
    name_list.append(filename)
    case_list.append(case)
# ftiaxnei ena dataframe me to damage percentage kai prosthetei to index number kai kanei sort basei autou 
dmg_data = pd.DataFrame({'dmg':dmg_list,'damage_file_name':name_list,'caseStudey':case_list})
correct_order = dmg_data['caseStudey']
dmg_data['dmg_index_number'] = [int(i.split('_')[-1]) for i in dmg_data['damage_file_name']]
dmg_data = dmg_data.sort_values(by=['caseStudey'])




odd_layer =['DamageLayer1', 'DamageLayer3', 'DamageLayer5'] ###['DamageLayer1', 'DamageLayer2', 'DamageLayer3', 'DamageLayer4', 'DamageLayer5']

for layer in odd_layer:
    dtotal_list = []
    i = 0
    for path in dmg_data['damage_file_name']:
        path = path + '.csv'
        dataframe = pd.read_csv(path,sep=' |,', engine='python')
        DL = dataframe[layer]
        df = DL[0]
        dm = DL[1]
        if df==0 and dm ==0:
            dtotal_list.append('clean')
            i = i+1
        elif df==0:
            dtotal_list.append('dm')
        elif dm ==0:
            dtotal_list.append('df')
        else:
            dtotal_list.append('df&dm')

    if layer == 'DamageLayer1':
        DL1 = pd.DataFrame({'Layer_1': dtotal_list})
    if layer == 'DamageLayer3':
        DL3 = pd.DataFrame({'Layer_3': dtotal_list})
    if layer == 'DamageLayer5':
        DL5 = pd.DataFrame({'Layer_5': dtotal_list})
even_layer = [ 'DamageLayer2',  'DamageLayer4']

for layer in even_layer:
    dd_list = []
    for path in dmg_data['damage_file_name']:
        path = path + '.csv'
        dataframe = pd.read_csv(path,sep=' |,', engine='python')
        DL = dataframe[layer]
        dd = DL[0]

        if dd ==0:
            dd_list.append('clean')
        else :
            dd_list.append('dd')

    if layer == 'DamageLayer2':
        DL2 = pd.DataFrame({'Layer_2': dd_list})
    if layer == 'DamageLayer4':
        DL4 = pd.DataFrame({'Layer_4': dd_list})


layer_damage = layer_damage.assign(**DL1,**DL2,**DL3,**DL4,**DL5)

dm_df_dd_list = []

for i in range(0,len(layer_damage)):
    if layer_damage['Layer_1'][i] == 'clean' and layer_damage['Layer_3'][i] == 'clean' and layer_damage['Layer_5'][i] == 'clean':
        if layer_damage['Layer_2'][i] == 'dd' or layer_damage['Layer_4'][i] == 'dd':
            dm_df_dd_list.append('dd')
        else :
            dm_df_dd_list.append('clean')
    elif layer_damage['Layer_1'][i] != 'clean' or layer_damage['Layer_3'][i] != 'clean' or layer_damage['Layer_5'][i] != 'clean':
        if layer_damage['Layer_2'][i] == 'dd' or layer_damage['Layer_4'][i] == 'dd':
            dm_df_dd_list.append('ola')
        else :
            if layer_damage['Layer_1'][i] == 'df' or layer_damage['Layer_3'][i] == 'df' or layer_damage['Layer_5'][i] == 'df':
                dm_df_dd_list.append('df')
            else:
                dm_df_dd_list.append('dm')



layer_damage['total_damage_per_layer'] = dm_df_dd_list


df_counter = 0
dm_counter = 0 
other_counter = 0
for sample in layer_damage['Layer_1']:
    if sample == 'df':
        df_counter +=1
    elif sample == 'dm':
        dm_counter +=1
    elif sample == 'df&dm':
        other_counter +=1
