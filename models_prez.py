
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



##### kanw train me to original dataset
from y_set_creator_dmg_percentage import y_set_creator
from feature_vector import X,X_fft


X = X_fft #X,X_fft,sensor_mean
y = y_set_creator('Damage_percentage','regression')


scaler = StandardScaler()
X= scaler.fit_transform(X)


################ PCA ################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

pca = PCA()
pca.fit(X)
pca_data = pca.transform(X)



#The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
#plt.show()



################################################

#X = pca.components_[0]
#X = pd.DataFrame({'pc1':pca.components_[0],'pc2':pca.components_[1],'pc3':pca.components_[2]})

############ kalw diafora montela############

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.000000002,
              max_iter=100000,
              tol=1.496e-07,
              selection='random'
              )
#lasso.fit(X,y)

from sklearn.linear_model import ElasticNetCV

en = ElasticNetCV(
                cv= 100,
                eps=1e-3,
                l1_ratio=0.99,  
                max_iter=1000000,  
                tol=0.000001,  
                selection='cyclic'
                )
en.fit(X,y)


################################################


##### kanw test me times rek


from feature_vector_dokimes_sample_rek import X_dokimes,X_fft_dokimes
from y_dokimes import damage_data_df

X = X_fft_dokimes
X = scaler.fit_transform(X)




y_pred = en.predict(X)
print(y_pred)

y_true = [0.02,0.034,0.062,0.086,0.1
          ]


print('mape is')
print(mean_absolute_percentage_error(y_true,y_pred))
print('mae is')
print(mean_absolute_error(y_true,y_pred))
