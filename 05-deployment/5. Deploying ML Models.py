#!/usr/bin/env python
# coding: utf-8

# # Deploying Machine Learning Models

# ## 5.1 Overview
# 
# - we want to train a model in our jupyter notebook and save this model to a file (model.bin)
# - Then we want to load this model from a different process (web service called "churn service" which has this model inside)
# - this Service is to identify churning customers
# - Let's say we have another service "marketing service" that holds all information about the customers, and for one customer we want to check the churning probability. What the marketing service is doing is sending a request to the churn service with the required customer information. The churn service answer this request and returns the prediction. Based on that the marketing service can decide whether they want to send a promotional email with some discount.
# 
# What we want to cover in this chapter is the deploying part. Take the Jupyter notebook and take the model and save it. Then load this model by the churn web service. Lastly we want to see how to interact with this service.
# To bring the model into a web service we'll use flask (which is a framework for creating web services). Then we want to isolate the dependencies for this service in a way that they don't interfere with other services that we have on our machine. So we want to create a special environment for python dependencies - for that we'll use pipenv. Then we'll add another layer on top. This layer will be a layer with system dependencies - for that we'll use docker. After that we'll deploy this in a cloud. So we take the container and deploy it to AWS Elastic Beanstalk.

# ## 5.2 Saving and loading the model
# - Saving the model to pickle
# - Loading the model from pickle
# - Turning our notebook into a Python script
# 
# In the previous session we trained a model for predicting churn and evaluated it. Now let's deploy it.

# In[7]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[8]:


# Data preparation

df = pd.read_csv('data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[9]:


# Data splitting

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[10]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[11]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


# In[12]:


def predict(df, dv, model):
     dicts = df[categorical + numerical].to_dict(orient='records')

     X = dv.fit_transform(dicts)
     y_pred = model.predict_proba(X)[:,1]

     return y_pred


# In[13]:


# Setting model parameters
# We're using 5 splits for evaluating our model using cross validation.

C = 1.0
n_splits = 5


# In[14]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)  

    
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# Output: C=1.0 0.841 +- 0.008


# In[15]:


scores

# Output: 
# [0.8438508214866044,
# 0.8450763971659383,
# 0.8327513546056594,
# 0.8301724275756219,
# 0.8521461516739357]


# In[16]:


# Train our final model

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)
auc

# Output: 0.8572386167896259


# Until now the model still lives in our notebook. So we cannot just take this model and put it in a web service. Remember we want to put this model in the web service, that the marketing service can use it to score the customers.
# That means now we need to save this model in order to be able to load it later.

# ### Saving the model with Pickle
# For saving the model we'll use pickle, what is a built in library for saving Python objects.

# In[17]:


import pickle


# In[18]:


# Now we need to take our model and write it to a file
output_file = 'model_C=%s.bin' % C
output_file

# Output: 'model_C=1.0.bin'


# In[19]:


# Another way to do this, where we can directly refer to the variable c.
output_file = f'model_C={C}.bin'
output_file

# Output: 'model_C=1.0.bin'


# In[20]:


# Now we want to create a file with that file name
# 'wb' means Write Binary
f_out = open(output_file, 'wb')
# We need to save DictVectorizer and the model as well, because with just the model we'll not be able to
# translate a customer into a feature matrix
pickle.dump((dv, model), f_out)
# Closing the file is pretty important otherwise we cannot be sure that this file really has the content.
f_out.close()


# In[21]:


# To not accidentially forgetting to close the file we can use the with statement, which makes sure that the file is closed all 
# the time so it automatically closes the file.
# Èverything we do inside the with statement the file is still open. Once we go out this statement, the file is automatically closed.
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# ### Loading the model with Pickle
# 

# In[1]:


import pickle


# In[2]:


model_file = 'model_C=1.0.bin'


# In[4]:


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

dv, model
# Output: (DictVectorizer(sparse=False), LogisticRegression(max_iter=1000))


# In[ ]:





# In[5]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[8]:


# Now let's turn this customer into a feature matrix
# The DictVectorizer expects a list of dictionaries, that's why we create a list with one customer.
X = dv.transform([customer])
X

# Output: 
# array([[ 1.  ,  0.  ,  0.  ,  1.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.  ,
#         0.  ,  1.  ,  0.  ,  0.  , 29.85,  0.  ,  1.  ,  0.  ,  0.  ,
#         0.  ,  1.  ,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  1.  ,
#         0.  ,  0.  ,  1.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.  ,  0.  ,
#         0.  ,  1.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.  , 29.85]])


# In[10]:


# Use predict function to get the probability that this particular customer is going to churn.

model.predict_proba(X)
# Output: array([[0.36364158, 0.63635842]])

# We're interested in the second element, so we need to set the row=0 and column=1
model.predict_proba(X)[0,1]
# Output: 0.6363584152758612


# To download the Jupyter Notebook file as Python file --> click "File" --> "Download as" --> Python (.py)

# In[ ]:




