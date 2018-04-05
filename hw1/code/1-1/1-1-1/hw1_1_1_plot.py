import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

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

plt.switch_backend('agg')
#plot f1 loss
plt.figure(figsize=(10,5))
plt.yscale('log')

with open('./history/history_shallow_f1','rb') as pickle_file:
    content = pickle.load(pickle_file)
plt.plot(content["loss"][0:30000])

with open('./history/history_medium_f1','rb') as pickle_file:
    content = pickle.load(pickle_file)
plt.plot(content["loss"][0:30000])
    
with open('./history/history_deep_f1','rb') as pickle_file:
    content = pickle.load(pickle_file)
plt.plot(content["loss"][0:30000])

plt.legend(["shallow_model_loss", "medium_model_loss","deep_model_loss"], loc="upper right")
plt.savefig("f1_loss.png")

#plot f2 loss
plt.figure(figsize=(10,5))
plt.yscale('log')

with open('./history/history_shallow_f2','rb') as pickle_file:
    content = pickle.load(pickle_file)
plt.plot(content["loss"][0:30000])

with open('./history/history_medium_f2','rb') as pickle_file:
    content = pickle.load(pickle_file)
plt.plot(content["loss"][0:30000])
    
with open('./history/history_deep_f2','rb') as pickle_file:
    content = pickle.load(pickle_file)
plt.plot(content["loss"][0:30000])

plt.legend(["shallow_model_loss", "medium_model_loss","deep_model_loss"], loc="upper right")
plt.savefig("f2_loss.png")

#plot f1 fit curve
x_ground_truth,y_ground_truth,x_train,y_train = create_training_data(f1)

plt.figure(figsize=(10,5))
plt.plot(x_ground_truth,y_ground_truth)

predict = np.load('./predict/predict_shallow_f1.npy')
plt.plot(x_ground_truth,predict)

predict = np.load('./predict/predict_medium_f1.npy')
plt.plot(x_ground_truth,predict)

predict = np.load('./predict/predict_deep_f1.npy')
plt.plot(x_ground_truth,predict)

plt.legend(["sin(exp(xπ))","shallow_model","medium_model","deep_model"], loc="lower left")
plt.savefig("f1_fit.png")

#plot f2 fit curve
x_ground_truth,y_ground_truth,x_train,y_train = create_training_data(f2)

plt.figure(figsize=(10,5))
plt.plot(x_ground_truth,y_ground_truth)

predict = np.load('./predict/predict_shallow_f2.npy')
plt.plot(x_ground_truth,predict)

predict = np.load('./predict/predict_medium_f2.npy')
plt.plot(x_ground_truth,predict)

predict = np.load('./predict/predict_deep_f2.npy')
plt.plot(x_ground_truth,predict)

plt.legend(["cos(x^3)","shallow_model","medium_model","deep_model"], loc="lower left")
plt.savefig("f2_fit.png")