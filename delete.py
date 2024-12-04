import matplotlib.pyplot as plt
import numpy as np

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