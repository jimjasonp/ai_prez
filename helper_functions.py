import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error

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
    plt.show()

def res_plot(model_list,min,mid,max,name_list):

    X_axis = np.arange(len(model_list)) 

    plt.bar(X_axis - 0.25, min, 0.2, label = '75 training samples')
    
    for index, value in enumerate(min):
        plt.text(value, index,str(value))
    
    plt.bar(X_axis , mid, 0.2, label = '112 training samples')
    for index, value in enumerate(mid):
        plt.text(value, index,str(value))
    
    plt.bar(X_axis + 0.25 , max, 0.2, label = '225 training samples')
    for index, value in enumerate(max):
        plt.text(value, index,str(value))
    
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