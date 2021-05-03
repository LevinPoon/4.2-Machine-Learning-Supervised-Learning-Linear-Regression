#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

dataset=pd.read_csv('StudentMarks.csv')

print(str(dataset.shape) + "rows & columns")
print("")

print("The 1st 5 row informations as below")
print(dataset.head()) 

print("\n")
print(dataset.describe())

dataset.plot(x='Hours',y='Scores', style="*")
plt.title('Student Marks Prediction')
plt.xlabel('Hours')
plt.ylabel('Percentage marks')
plt.show()

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

print("The Intercept value is: " + str(regressor.intercept_))

print("The Slope value is: " + str(regressor.coef_))

y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual': Y_test,'Predicted': y_pred})
df


# In[ ]:




