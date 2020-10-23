#!/usr/bin/env python
# coding: utf-8

# In[131]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[132]:


#import libraries
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding,Activation,Dropout
from tensorflow.keras.layers import Conv1D,MaxPooling1D,GlobalAveragePooling1D
import numpy as np
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split


# In[174]:


#Reading csv file
df=pd.read_csv('\\Users\\Mainak\\Desktop\\spyder\\twitter16.csv',encoding='latin1',header=None)
df.head()


# In[175]:


#Data preprocessing
df=df[[5,0]]
df.columns=['tweet','sentiment']


# In[177]:


#Sampling 25% of data randomly
df=df.sample(frac=0.1, replace=True, random_state=1)
df.shape


# In[207]:


df['sentiment'].apply(lambda x:1 if x==4 else 0)
text=df['tweet'].tolist()


# In[208]:


y=df['sentiment']


# In[209]:


#Fitting tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(text)


# In[210]:


vocab_size=len(tokenizer.word_index)+1
vocab_size


# In[211]:


#Converting texts to sequences
encoded_text=tokenizer.texts_to_sequences(text)


# In[212]:


encoded_text[:3]


# In[213]:


#Padding the text sequences to a common length
max_length=120
X=pad_sequences(encoded_text,maxlen=max_length,padding='post')


# In[214]:


print(X.shape)
print(X[0])


# In[215]:


#Creating train test datasets
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)


# In[216]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print("---------------------")
print(y_train.value_counts())
print("---------------------")
print(y_test.value_counts())


# In[217]:



vec_size=300
model=Sequential()
model.add(Embedding(vocab_size,vec_size,input_length=max_length))
model.add(Conv1D(64,8,activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.summary()


# In[218]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))')


# In[ ]:





# In[ ]:


def get_encoded(x):
    x=tokenizer.texts_to_sequences(x)
    x=pad_sequences(x,maxlen=max_length,padding='post')
    return x


# In[ ]:


x=["worst services.will not come again"]
model.predict_classes(get_encoded(x)) 


# In[206]:


df['sentiment'].apply(


# In[ ]:




