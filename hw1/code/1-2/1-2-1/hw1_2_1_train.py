from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import TensorBoard,CSVLogger,ModelCheckpoint
from keras.callbacks import Callback
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
    x_ground_truth = np.linspace(-range_len, range_len, 100)
    y_ground_truth = f(x_ground_truth)
    x_train = np.linspace(-range_len, range_len, 500000)
    y_train = f(x_train)
    return x_ground_truth,y_ground_truth,x_train,y_train

def model_shallow():
    model = Sequential()
    
    model.add(Dense(712,input_dim=1, activation = 'relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def model_medium():
    model = Sequential()
    
    model.add(Dense(25,input_dim=1, activation = 'relu'))
    model.add(Dense(40, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def model_deep():
    model = Sequential()
    
    model.add(Dense(5,input_dim=1, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(15, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(15, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(5, activation = 'relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def train_and_save(build_model,model_name,epoch):
    model_checker = ModelCheckpoint('./weight_' + model_name + '/' + model_name + '_weights.{epoch:04d}.hdf5', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=3)
    
    model = build_model()
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=256,callbacks=[model_checker])
    y_predict = model.predict(x_ground_truth)


# create training data
x_ground_truth,y_ground_truth,x_train,y_train = create_training_data(f1)

# train
for i in range(1,9):
    train_and_save(model_deep,str(i),750)

# merge one training weights to dimension
all_train_weights = []

for train_time in range(1,9):
    for epoch in range(2,750,3):
        # 讀其中一次訓練的其中一個epcoh權重
        model = model_deep()
        file_path = './weight_%d/%d_weights.%04d.hdf5' %(train_time,train_time,epoch)
        model.load_weights(file_path)
        
        # 將此權重轉成一維
        w = model.get_weights()
        one_dim_w = []
        for i in range(len(w)):
            # 先全部拉成一維存起來
            one_dim_w.append(w[i].reshape(-1,1))
        
        # 將其中一次訓練的其中一個epcoh的所有權重合併成一個一維array
        merged_w = np.concatenate(one_dim_w,axis=0)
        all_train_weights.append(merged_w)
        print(file_path)

# 將每次訓練的每個epcoh權重合併成一個matrix
weights_matrix = np.concatenate(all_train_weights,axis=1)
print(weights_matrix.shape)
np.save("./weights_matrix.npy",weights_matrix)