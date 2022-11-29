import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as dsplit
from sklearn.linear_model import LinearRegression as lr
import seaborn as sb
from sklearn import metrics as mt
import tkinter as tk
# Reading of DataSet
df=pd.read_csv(r"C:\Users\donal_yqj9nme\OneDrive\Desktop\Ecommerce_Customers.csv.xls")
x,y=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']],df[['Yearly Amount Spent']]
# Splitting of DataSet into TrainingSet and TestingSet
x_tr,x_t,y_tr,y_t=dsplit(x,y,test_size=0.3,random_state=101)
# Fitting of Model
reg=lr()
reg.fit(x_tr,y_tr)
# Predictions from Model
pre=reg.predict(x_t)
cdf=pd.DataFrame(reg.coef_[0],x.columns,columns=['coeff'])
# Finding Error Rate
print(cdf)
print(mt.mean_absolute_error(y_t,pre),mt.mean_squared_error(y_t,pre),np.sqrt(mt.mean_squared_error(y_t,pre)),sep='\n')
# Plotting Graphs on Error Rate
plt.scatter(pre,y_t,cmap='')
sb.displot(y_t-pre)
plt.show()