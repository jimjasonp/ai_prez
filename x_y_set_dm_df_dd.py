from y_set_df_dm_dd import dm_df_dd_list
from feature_vector_test import sensor2_vector,sensor3_vector,sensor4_vector
import numpy as np
import pandas as pd



df = pd.DataFrame({'defects':dm_df_dd_list})


index_list = []

for i in range (0, len(df)):
    if df['defects'][i] != 'df' and df['defects'][i] != 'dm' and df['defects'][i] != 'dd':
        df = df.drop([i])
        index_list.append(i)

df = df.drop([119])
df = df.drop([127])

del sensor2_vector[106:]
del sensor2_vector[73]
del sensor2_vector[0]


del sensor3_vector[106:]
del sensor3_vector[73]
del sensor3_vector[0]


del sensor4_vector[106:]
del sensor4_vector[73]
del sensor4_vector[0]



y = df
X = np.concatenate((sensor2_vector,sensor3_vector,sensor4_vector),axis=1)