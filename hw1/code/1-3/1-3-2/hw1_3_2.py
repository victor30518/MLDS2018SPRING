
# coding: utf-8

# In[1]:
import numpy as np

# In[2]:
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 20
neuron=16


img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# In[3]:
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[4]:
model = Sequential()
model.add(Dense(neuron, activation='relu', input_shape=(784,)))
model.add(Dense(neuron, activation='relu'))
model.add(Dense(neuron, activation='relu'))
model.add(Dense(neuron, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# In[5]:
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="Adam",
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
# In[10]:


import pickle
with open('./model'+str(neuron), 'wb') as file:
    pickle.dump(history.history, file)


# In[6]:
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[7]:
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel("train")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="best")
    plt.show()


# In[8]:
show_train_history(history, "acc","val_acc")
# In[9]:
show_train_history(history, "loss","val_loss")




