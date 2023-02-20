#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[ ]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[ ]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[ ]:


df['loan_status'].value_counts()


# In[ ]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[ ]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins=np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins=np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[ ]:


df['weekend']= df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[ ]:



df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[ ]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[ ]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[ ]:


df[['Principal','terms','age','Gender','education']].head()


# In[ ]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[ ]:


X = Feature
X[0:5]


# In[ ]:


# What are our lables?

y = df['loan_status'].values
y[0:5]


# In[ ]:


X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

