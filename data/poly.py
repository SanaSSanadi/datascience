#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[23]:


df=pd.read_excel("sal_exp.xlsx")


# In[24]:


plt.scatter(df['Experience'],df['Salary'])


# In[25]:


poly=PolynomialFeatures(degree=2)
X=poly.fit_transform(df[['Experience']])


# 

# In[26]:


model=LinearRegression()


# In[27]:


model.fit(X,df['Salary'])


# In[28]:


plt.scatter(df['Experience'],df['Salary'])
y_pred=model.predict(X)


# In[29]:


model.fit(X,df['Salary'])


# In[30]:


X=poly.fit_transform([[133.33]])


# In[31]:


y=model.predict(X)


# In[32]:


query=[[15]]
X_query=poly.transform(query)


# In[33]:


model.predict(X_query)


# In[34]:


pickle.dump(model,open('model1.pkl','wb'))


# In[35]:


pickle.dump(poly,open('poly1.pkl','wb'))


# In[36]:


obj=pickle.load(open('model1.pkl','rb'))


# In[ ]:





# In[ ]:




