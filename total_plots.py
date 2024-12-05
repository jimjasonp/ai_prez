import matplotlib.pyplot as plt
import numpy as np




def res_plot(model_list,min,mid,max):


    X = ['Linear Regression','Elastic Net','Ridge Regression', 'MLP'] 

    min = [9.2 , 15.0, 9.0, 16.0]
    mid = [6.3 , 16.2, 6.1, 10.2]
    max = [8.5 , 13.7, 8, 14.0]

    X_axis = np.arange(len(model_list)) 

    plt.bar(X_axis - 0.25, min, 0.2, label = '90 training samples') 
    plt.bar(X_axis , mid, 0.2, label = '150 training samples') 
    plt.bar(X_axis + 0.25 , max, 0.2, label = '210 training samples') 
    
    plt.xticks(X_axis, model_list) 
    plt.xlabel("Models")
    plt.ylabel("Mean absolute Percentage error")
    plt.title(f"MAPE of models with different training sizes ")
    plt.legend() 
    plt.show()

X = ['Linear Regression','Elastic Net','Ridge Regression', 'MLP'] 

min = [9.2 , 15.0, 9.0, 16.0]
mid = [6.3 , 16.2, 6.1, 10.2]
max = [8.5 , 13.7, 8, 14.0]

X_axis = np.arange(len(X)) 

plt.bar(X_axis - 0.25, min, 0.2, label = '90 training samples') 
plt.bar(X_axis , mid, 0.2, label = '150 training samples') 
plt.bar(X_axis + 0.25 , max, 0.2, label = '210 training samples') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Models")
plt.ylabel("Mean absolute Percentage error")
plt.title(f"MAPE of models with different training sizes ")
plt.legend() 
plt.show()






######### good models ################
data = {'Linear Regression':model_run(lr)[1], 
        'Ridge':model_run(rr)[1],
        'Elastic Net':model_run(en)[1],
        
        }

model_names  = list(data.keys())
mape = list(data.values())

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(model_names, mape, color ='maroon', 
        width = 0.4)

samples = len(X_train)
plt.xlabel("Models")
plt.ylabel("Mean absolute Percentage error")
plt.title(f"Mean absolute percentage error of models with number of samples used = {samples}")
plt.show()
################################################
data = {'MLP':model_run(mlp)[1], 
        'Decision Trees':model_run(dt)[1],
        'Random Forest':model_run(rf)[1]
        }

model_names  = list(data.keys())
mape = list(data.values())

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(model_names, mape, color ='maroon', 
        width = 0.4)

samples = len(X_train)
plt.xlabel("Models")
plt.ylabel("Mean absolute Percentage error")
plt.title(f"Mean absolute percentage error of models with number of samples used = {samples}")
plt.show()
################  bad models  ################


################################################
plt.plot(model_run(lr)[2],marker = 'o')
plt.plot(model_run(lr)[3],linestyle='dashed',marker = 'o')
plt.xlabel("sample")
plt.ylabel("y value")
plt.title(f" Predicted and true value of samples using Linear Regression")
plt.legend(["y_test", "y_pred"], loc="lower right")
#plt.show()

