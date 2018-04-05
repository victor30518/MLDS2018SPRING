
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pickle
import tensorflow as tf
import os
import numpy as np
from keras import backend as K
sess=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.095)))
K.set_session(sess)
np.random.seed(128)


# In[2]:


batch_size = 256
num_classes = 10
epochs = 100
data_augmentation = False
#num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_model_mid.h5'


# In[3]:


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[4]:


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(22, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[5]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

indices = np.random.permutation(y_train.shape[0])
x = x_train[indices]
y = y_train[indices] 
# xvalid = x[:int(len(x)*0.1)]
# yvalid = y[:int(len(y)*0.1)]
# xtrain = x[int(len(x)*0.1):]
# ytrain = y[int(len(y)*0.1):]
xtrain = x
ytrain = y


# In[6]:


history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
               validation_data=(x_test, y_test),
              shuffle=True)


# In[11]:


with open('./mid.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# In[ ]:


with open('./mid.pickle', 'rb') as handle:
    history = pickle.load(handle)


# In[9]:


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

