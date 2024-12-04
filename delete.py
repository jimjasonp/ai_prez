import matplotlib.pyplot as plt
import numpy as np

X = ['Linear Regression','Elastic Net','Ridge Regression'] 

lr = [9.2 ,  6.3, 8.1 ] 
en = [15.9 , 16.2, 13.7] 
rr = [9.0 , 6.1 , 7.9]

min = [9.2 , 15.9, 9.0]
mid = [6.3 , 16.2, 6.1]
max = [8.1 , 13.7, 7.9]

X_axis = np.arange(len(X)) 

plt.bar(X_axis - 0.25, min, 0.2, label = '57 training samples') 
plt.bar(X_axis , mid, 0.2, label = '117 training samples') 
plt.bar(X_axis + 0.25 , max, 0.2, label = '177 training samples') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Models")
plt.ylabel("Mean absolute Percentage error")
plt.title(f"MAPE of models with different training sizes ")
plt.legend() 
plt.show()