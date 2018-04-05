from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os

# # GPU設定
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #1080
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

def f1(x):
    return np.sin(np.exp(x*np.pi))

def f2(x):
    return np.cos(10*x**3)

def create_training_data(f):
    np.random.seed(5)
    range_len = 1
    x_ground_truth = np.linspace(-range_len, range_len, 50000)
    y_ground_truth = f(x_ground_truth)
    x_train = np.random.uniform(-range_len,range_len,1000)
    y_train = f(x_train)
    return x_ground_truth,y_ground_truth,x_train,y_train

def model_shallow():
    model = Sequential()
    
    model.add(Dense(30,input_dim=1, activation = 'relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def model_medium():
    model = Sequential()
    
    model.add(Dense(4,input_dim=1, activation = 'relu'))
    model.add(Dense(6, activation = 'relu'))
    model.add(Dense(4, activation = 'relu'))
    model.add(Dense(4, activation = 'relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def model_deep():
    model = Sequential()
    
    model.add(Dense(4,input_dim=1, activation = 'relu'))
    model.add(Dense(4, activation = 'relu'))
    model.add(Dense(3, activation = 'relu'))
    model.add(Dense(3, activation = 'relu'))
    model.add(Dense(3, activation = 'relu'))
    model.add(Dense(3, activation = 'relu'))
    model.add(Dense(2, activation = 'relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def train_and_save(build_model,model_name,epoch):
    model = build_model()
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=64)
    y_predict = model.predict(x_ground_truth)
    
    np.save('./predict/predict_' + model_name + ".npy",y_predict)
    
    with open('./history/history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# train
x_ground_truth,y_ground_truth,x_train,y_train = create_training_data(f1)
train_and_save(model_shallow,"shallow_f1",30000)
train_and_save(model_medium,"medium_f1",30000)
train_and_save(model_deep,"deep_f1",30000)

x_ground_truth,y_ground_truth,x_train,y_train = create_training_data(f2)
train_and_save(model_shallow,"shallow_f2",30000)
train_and_save(model_medium,"medium_f2",30000)
train_and_save(model_deep,"deep_f2",30000)