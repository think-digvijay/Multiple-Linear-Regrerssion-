#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


# In[2]:


bh_data = load_boston()


# In[3]:


print(bh_data.keys())


# In[4]:


boston = pd.DataFrame(bh_data.data, columns=bh_data.feature_names)


# In[5]:


print(bh_data.DESCR)


# In[7]:


boston['MEDV'] = bh_data.target


# In[9]:


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT','RM'])
y = boston['MEDV']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=9)


# In[11]:


lin_reg_mod = LinearRegression()

lin_reg_mod.fit(X_train, y_train)


# In[12]:


pred = lin_reg_mod.predict(X_test)

test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

test_set_r2 = r2_score(y_test, pred)


# In[13]:


print(test_set_rmse)
print(test_set_r2)


# In[ ]:




