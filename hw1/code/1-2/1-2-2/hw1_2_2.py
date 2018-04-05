
# coding: utf-8

# # load MNIST data

# In[470]:


import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import pylab

# # define model

# In[471]:


import tensorflow as tf


# In[472]:


x = tf.placeholder("float", [None, 784]) #input


# In[473]:


h1 = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu, use_bias=False)
h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu, use_bias=False)
output = tf.layers.dense(inputs=h2, units=10, activation=None, use_bias=False)


# In[474]:


y_ = tf.placeholder("float", [None, 10]) #ouput


# In[475]:


loss = tf.losses.softmax_cross_entropy(logits = output, onehot_labels= y_) # loss fun ction


# In[476]:


optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)


# # calculate gradient

# In[477]:


grads_and_vars = optimizer.compute_gradients(loss)
grads, _ = list(zip(*grads_and_vars))
norms = tf.global_norm(grads)


# # accuracy

# In[478]:


correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))


# In[479]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[480]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[481]:


norms_ls=[] #store gradients norm per step
loss_ls=[] #store loss per step


# In[482]:


for i in range(15000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, norm, l= sess.run([train_step, norms, loss], feed_dict={x: batch_xs, y_: batch_ys})
    norms_ls.append(norm)
    loss_ls.append(l)
    
    if i % 600 == 0:
        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
        print("step ", i, ", loss: ", train_loss, ", acc: ", train_acc, ", val_loss: ", val_loss, ", val_acc: ", val_acc)

        


# In[483]:


print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[491]:


sess.close()


# In[532]:




# In[551]:


def plot_list(train_history, label, train_history2, label2):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    pylab.ylim(-2,4)
    ax1.plot(train_history, 'b')
    ax1.legend([label], loc = 'upper center')
    ax1.set_ylabel(label, color="blue")
    plt.xlabel('step')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(train_history2, 'r')
    ax2.set_ylabel(label2, color="red")
    ax2.legend([label2], loc = 'upper right')
    plt.title('gradient norms/loss')
    plt.show()
    plt.savefig('gredient_norm.png')


# In[552]:


plot_list(norms_ls, "gredient norm", loss_ls, "loss")

