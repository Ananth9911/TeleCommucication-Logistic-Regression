#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 


# In[6]:


churn_csv=pd.read_csv("Tele.csv")
churn_csv.head()


# In[7]:


churn_csv=churn_csv[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]
churn_csv['churn']=churn_csv['churn'].astype(int)
churn_csv[0:5]


# In[8]:


x=np.asarray(churn_csv[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']])
x[0:5]


# In[9]:


y=np.asarray(churn_csv['churn'])
y[0:5]


# In[10]:


x=preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print("Training Set Of X ",x_train.shape,y_train.shape)


# In[14]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
LR


# In[16]:


pred=LR.predict(x_test)


# In[17]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test,pred)


# In[ ]:




