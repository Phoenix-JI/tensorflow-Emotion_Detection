#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
import os


# In[2]:


def getModel(INPUT_SHAPE = (48,48,1)):

    inp = keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(inp)  
    pool1 = keras.layers.MaxPooling2D(pool_size=2)(conv1)
    norm1 = keras.layers.BatchNormalization()(pool1)
    drop1 = keras.layers.Dropout(0.4)(norm1)

    conv2 = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(drop1)
    pool2 = keras.layers.MaxPooling2D(pool_size=2)(conv2)
    norm2 = keras.layers.BatchNormalization()(pool2)
    drop2 = keras.layers.Dropout(0.4)(norm2)

    conv3 = keras.layers.Conv2D(128, kernel_size=3,activation='relu', padding='same')(drop2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    norm3 = keras.layers.BatchNormalization()(pool3)
    drop3 = keras.layers.Dropout(0.4)(norm3)

    conv4 = keras.layers.Conv2D(256, kernel_size=3,activation='relu', padding='same')(drop3)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    norm4 = keras.layers.BatchNormalization()(pool4)
    drop4 = keras.layers.Dropout(0.4)(norm4)

    conv5 = keras.layers.Conv2D(512, kernel_size=3,activation='relu', padding='same')(drop4)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)
    norm5 = keras.layers.BatchNormalization()(pool5)
    drop5 = keras.layers.Dropout(0.4)(norm5)

    flat = keras.layers.Flatten()(drop5)  #Flatten the matrix to get it ready for dense.
    #flat = keras.layers.GlobalAveragePooling2D()(drop4)

    hidden1 = keras.layers.Dense(1024, activation='relu')(flat)
    drop4 = keras.layers.Dropout(0.4)(hidden1)

    out = keras.layers.Dense(7, activation='softmax')(drop4)   

    model = keras.Model(inputs=inp, outputs=out)

    return model


# In[3]:


model = getModel()


# In[4]:


model.load_weights('/Users/phxji/Desktop/2D_EmoClass.h5')


# In[5]:


emotion_dict = {0: "Angry", 2: "Fearful", 3: "Happy", 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}


# In[6]:


import random

Department = ['D1','D2','D3','D4']
Time = ['T1','T2','T3','T4']
Project = ['P1','P2','P3','P4']
Emotion = []
TotalRecord = []


# In[ ]:


cap = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('/Users/phxji/Desktop/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    print(np.array(faces).shape)

    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        
        img = cv2.resize(roi_gray, (48, 48))
                         
        cropped_img = np.expand_dims(np.expand_dims(img,-1),0)
        
        #print(cropped_img.shape)
        prediction = model.predict(cropped_img)
        index = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[index], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        Emotion.append(index)
        Record = []
        #Record.append(maxindex)
        ramindex = random.randint(0, 3)
        Record.append(Department[ramindex])
        ramindex = random.randint(0, 3)
        Record.append(Time[ramindex])
        ramindex = random.randint(0, 3)
        Record.append(Project[ramindex])
        Record=np.array(Record)
        
        TotalRecord.append(Record)
        
        
    cv2.imshow('Emotion Decection', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


print(len(TotalRecord))
print(len(Emotion))


# # Test Classification to give task

# In[9]:


TotalRecord = np.array(TotalRecord.copy())
Emotion = np.array(Emotion.copy())
TotalRecord = transform(TotalRecord.copy())
TotalRecord = TotalRecord.astype(np.int)


# In[ ]:





# In[10]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(TotalRecord, Emotion)


# In[11]:


X_test = ['D4','T2','P4']
X_test = transform(X_test)
print(X_test)
X_Te = X_test.astype(np.int)
test = []
test.append(X_Te)
te = np.array(test)
svclassifier.predict(te)


# In[7]:


def transform(X_test):
    
    X_test = np.where((X_test=='D1')|(X_test=='T1')|(X_test=='P1'),1,X_test)
    X_test = np.where((X_test=='D2')|(X_test=='T2')|(X_test=='P2'),2,X_test)
    X_test = np.where((X_test=='D3')|(X_test=='T3')|(X_test=='P3'),3,X_test)
    X_test = np.where((X_test=='D4')|(X_test=='T4')|(X_test=='P4'),4,X_test)
    
    return X_test


# In[ ]:




