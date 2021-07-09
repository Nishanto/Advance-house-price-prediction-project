#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

## for feature slection

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[2]:


dataset=pd.read_csv('X_train.csv')


# In[3]:


dataset.head()


# In[4]:


## Capture the dependent feature
y_train=dataset[['SalePrice']]


# In[5]:


## drop dependent feature from dataset
X_train=dataset.drop(['Id','SalePrice'],axis=1)


# In[6]:


### Apply Feature Selection
# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)


# In[7]:


feature_sel_model.get_support()


# In[8]:


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))


# In[9]:


selected_feat


# In[10]:


X_train=X_train[selected_feat]


# In[11]:


X_train.head()


# In[ ]:




