import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(1)
np.random.seed(1)
import input_data

loss_list = np.load('train_loss.npy')
loss_list2 = np.load('val_loss.npy')
samples = np.linspace(0, 1., 2000)
plt.figure()
plt.plot(samples, loss_list, color='blue')

# plt.xlabel('alpha')
# plt.ylabel('cross entropy loss')
# plt.title('training')
# plt.show()
# plt.close()
plt.hold(True)
# plt.figure()
plt.plot(samples, loss_list2,color='red')
plt.legend(['train_loss','val_loss'])
plt.xlabel('alpha')
plt.ylabel('cross entropy loss')
# plt.title('validation')
plt.show()
plt.close()

