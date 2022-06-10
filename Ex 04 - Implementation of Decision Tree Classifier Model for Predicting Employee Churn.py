#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd


# In[17]:


data=pd.read_csv("Employee.csv")


# In[18]:


data.head()


# In[19]:


data.info()


# In[20]:


data.isnull().sum()


# In[21]:


data["left"].value_counts()


# In[22]:


from sklearn.preprocessing import LabelEncoder


# In[23]:


le=LabelEncoder()


# In[24]:


data["salary"]=le.fit_transform(data["salary"])


# In[34]:


data.head()


# In[38]:


x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]


# In[39]:


x.head()


# In[40]:


y=data["left"]


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


dt=DecisionTreeClassifier(criterion="entropy")


# In[53]:


dt.fit(x_train,y_train)


# In[54]:


y_pred=dt.predict(x_test)


# In[55]:


from sklearn import metrics


# In[56]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[57]:


accuracy


# In[58]:


dt.predict([[0.5,0.8,9,260,6,0,1,2]])


# In[ ]:




