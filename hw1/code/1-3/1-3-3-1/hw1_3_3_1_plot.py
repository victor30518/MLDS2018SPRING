import matplotlib.pyplot as plt
import numpy as np

# load acc
loss_list = np.load('./interpolation/train_loss_1.npy')
loss_list2 = np.load('./interpolation/val_loss_1.npy')
samples = np.linspace(-1., 2., 10000)

# plot acc
plt.switch_backend('agg')
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.plot(samples, loss_list,"b")
ax1.plot(samples, loss_list2,"b--")

# 設定第一個y軸
ax1.set_ylabel('cross_entropy')
ax1.tick_params(axis='y', colors='blue')
ax1.set_xlabel('alpha')

# load loss
acc_list = np.load('./interpolation/train_acc_1.npy')
acc_list2 = np.load('./interpolation/val_acc_1.npy')

ax2 = ax1.twinx()
ax2.plot(samples, acc_list, 'r')
ax2.plot(samples, acc_list2, 'r--')
ax2.tick_params(axis='y', colors='red')

# 設定第二個y軸
ax2.legend(['train','test'],loc = "upper center")
# ax2.set_xlim([0, np.e])
ax2.set_ylabel('accuracy')

plt.savefig("interpolation_1.png")



# load acc
loss_list = np.load('./interpolation/train_loss_2.npy')
loss_list2 = np.load('./interpolation/val_loss_2.npy')
samples = np.linspace(-1., 2., 3000)

# plot acc
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.plot(samples, loss_list,"b")
ax1.plot(samples, loss_list2,"b--")

# 設定第一個y軸
ax1.set_ylabel('cross_entropy')
ax1.tick_params(axis='y', colors='blue')
ax1.set_xlabel('alpha')

# load loss
acc_list = np.load('./interpolation/train_acc_2.npy')
acc_list2 = np.load('./interpolation/val_acc_2.npy')

ax2 = ax1.twinx()
ax2.plot(samples, acc_list, 'r')
ax2.plot(samples, acc_list2, 'r--')
ax2.tick_params(axis='y', colors='red')

# 設定第二個y軸
ax2.legend(['train','test'],loc = "upper center")
# ax2.set_xlim([0, np.e])
ax2.set_ylabel('accuracy')

plt.savefig("interpolation_2.png")