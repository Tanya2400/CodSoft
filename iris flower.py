#!/usr/bin/env python
# coding: utf-8

# # TASK 2
# IRIS FLOWER CLASSIFICATION

# In[54]:


# Importing Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[55]:


data = pd.read_csv('downloads/IRIS.csv')


# In[56]:


data


# In[57]:


data.shape


# In[58]:


data.describe()


# In[59]:


data.groupby('species').mean()


# In[60]:


#Exploratory Data Analysis (EDA)
sns.heatmap(data.corr(), cmap=sns.cubehelix_palette(as_cmap=True))
plt.show()


# In[61]:


#Visualization
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=data, palette="flare")
plt.show()


# In[62]:


sns.displot(data=data.drop(['species'], axis=1))
plt.show()


# In[63]:


data.plot.hist(subplots=True,edgecolor='black', layout=(2,2), figsize=(8, 8), bins=20)
plt.show()


# In[64]:


g = sns.FacetGrid(data, col='species')
g = g.map(sns.kdeplot, 'sepal_length', multiple="stack")


# In[65]:


sns.pairplot(data, hue="species")


# In[66]:


data.hist(color= 'pink',edgecolor='black',figsize=(8,8))
plt.show()


# In[67]:


data.corr().style.background_gradient(cmap='cool').set_precision(2)


# In[68]:


#Training and testing the model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

x = data.drop('species', axis=1)
y= data.species

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)


# In[69]:


#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x, y)
y_pred = logreg.predict(x)
print(metrics.accuracy_score(y, y_pred))


# In[70]:


#Predicted Values
predict = logreg.predict(x_test)
# compare test and train data
compar = pd.DataFrame({'actual': y_test, 'predicted': predict})
compar = compar.reset_index(drop = True)
compar[:10]


# In[71]:


#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))


# In[ ]:




