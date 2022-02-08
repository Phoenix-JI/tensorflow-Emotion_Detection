#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Read Data Source File


# In[4]:


df = pd.read_csv('fer2013.csv')


# In[5]:


df.head()


# In[11]:


test = df['emotion'].value_counts()


# In[16]:


df['emotion'].value_counts()


# In[14]:


test.values


# In[15]:


plt.bar(test.index,test.values)


# In[5]:


emotionValue = df['emotion'].values
pixelValue = df['pixels'].values
print(emotionValue.shape)
print(pixelValue.shape)


# In[6]:


# Covert to dataset list and Label list


# In[57]:


EmoImg = np.zeros((48,48))
dataset=[]
Label=[]


# In[58]:


for index,pix in enumerate(pixelValue):
    
    if(emotionValue[index]==1):
        continue;
    
    EmoImg = np.zeros((48,48))
    
    pix = np.array(pix.split(' '))
    
    i=0
    j=0

    
    for piValue in pix:
        
        if(j == 48):
            #print(j)
            j=0
            i=i+1
            
        EmoImg[i][j] = int(piValue)
        j=j+1
    
    dataset.append(EmoImg)
    Label.append(emotionValue[index])


# In[59]:


len(dataset),len(Label)


# In[75]:


Label[60:80]


# In[77]:


plt.imshow(dataset[73],cmap='gray')


# In[62]:


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# In[63]:


a = np.array(Label)
unique, counts = np.unique(a, return_counts=True)
dict(zip(unique, counts))


# In[64]:


from sklearn.model_selection import train_test_split


# In[66]:


X_train,X_test,y_train,y_test = train_test_split(np.array(dataset),np.array(Label),test_size=0.2)


# In[67]:


np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)
np.save('X_test.npy',X_test)
np.save('y_test.npy',y_test)


# In[ ]:




