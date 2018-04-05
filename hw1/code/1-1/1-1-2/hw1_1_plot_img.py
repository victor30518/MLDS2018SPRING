
# coding: utf-8

# In[9]:


import pickle
import matplotlib.pyplot as plt


# In[11]:


with open('./deep.pickle', 'rb') as handle:
    deep_history = pickle.load(handle)


# In[13]:


with open('./mid.pickle', 'rb') as handle:
    mid_history = pickle.load(handle)


# In[14]:


with open('./shallow.pickle', 'rb') as handle:
    shallow_history = pickle.load(handle)


# In[18]:


plt.plot(deep_history['acc'])
plt.plot(deep_history['val_acc'])
plt.plot(mid_history['acc'])
plt.plot(mid_history['val_acc'])
plt.plot(shallow_history['acc'])
plt.plot(shallow_history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['deep_train_acc', 'deep_test_acc', 'mid_train_acc', 'mid_test_acc', 'shallow_train_acc', 'shallow_test_acc'], loc='lower right')
plt.show()



# In[19]:


# summarize history for loss
plt.plot(deep_history['loss'])
plt.plot(deep_history['val_loss'])
plt.plot(mid_history['loss'])
plt.plot(mid_history['val_loss'])
plt.plot(shallow_history['loss'])
plt.plot(shallow_history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['deep_train_loss', 'deep_test_loss', 'mid_train_loss', 'mid_test_loss', 'shallow_train_loss', 'shallow_test_loss'], loc='best')
plt.show()

