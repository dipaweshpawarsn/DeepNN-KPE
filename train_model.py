
# coding: utf-8

# In[ ]:

from config import *
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


candidate_column_nos=[i for i in range(1,309)]
#D:\\RevAnalyzer\\SNTPS-RevAnalyzer\\training.csv
dataset = loadtxt(file_for_training_deep_model, delimiter='@|@|@',usecols=tuple(candidate_column_nos),comments='!!!COMMENT!!!')


# In[8]:


X=dataset[:,300:307]
Y=dataset[:,307]


# In[38]:


model = Sequential()
model.add(Dense(6, input_dim=7, activation='sigmoid'))
model.add(Dense(6, input_dim=6, activation='sigmoid'))
model.add(Dense(6, input_dim=6, activation='sigmoid'))
model.add(Dense(6, input_dim=6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


# In[39]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[41]:


model.fit(X, Y, epochs=no_of_epochs, batch_size=batch_size)


# In[42]:


model.save(deep_model_path)

