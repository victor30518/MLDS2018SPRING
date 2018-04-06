
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
get_ipython().magic('matplotlib inline')


# In[2]:


#generate data
def f1(x):
    return np.sin(np.exp(x*np.pi))
    #return 5*x

def f2(x):
    return np.cos(10*x**3)

def create_training_data(f):
    np.random.seed(5)
    range_len = 1
    x_ground_truth = np.linspace(-range_len, range_len, 100)
    y_ground_truth = f(x_ground_truth)
    x_train = np.linspace(-range_len, range_len, 50000)
    y_train = f(x_train)
    return x_ground_truth,y_ground_truth,x_train,y_train

x_ground_truth,y_ground_truth,x_train,y_train = create_training_data(f1)


# In[3]:


x = tf.placeholder("float", [None, 1]) #input


# In[4]:


W_ls = []
W1 = tf.Variable(tf.random_normal([1, 40])) #generate a input_dim*output_dim matrix with normal random  weights
b1 = tf.Variable(tf.random_normal([1, 40]))
XWb1 = tf.matmul(x, W1)+b1
outputs1 = tf.nn.relu(XWb1)

W2 = tf.Variable(tf.random_normal([40, 40])) #generate a input_dim*output_dim matrix with normal random  weights
b2 = tf.Variable(tf.random_normal([1, 40]))
XWb2 = tf.matmul(outputs1, W2)+b2
outputs2 = tf.nn.relu(XWb2)

W3 = tf.Variable(tf.random_normal([40, 1])) #generate a input_dim*output_dim matrix with normal random  weights
b3 = tf.Variable(tf.random_normal([1, 1]))
XWb3 = tf.matmul(outputs2, W3)+b3
# outputs3 = tf.nn.relu(XWb3)

# W4 = tf.Variable(tf.random_normal([20, 1]))
# b4 = tf.Variable(tf.random_normal([1, 1]))
# XWb4 = tf.matmul(outputs3, W4)+b4
output = XWb3


# In[5]:


y_ = tf.placeholder("float", [None, 1]) #ouput


# In[6]:


# print(tf.trainable_variables())


# In[7]:


loss = tf.losses.mean_squared_error(labels=y_, predictions=output) # loss function


# In[8]:


#optimizer = tf.train.AdamOptimizer()

#optimizer = tf.train.GradientDescentOptimizer(0.0000000001)
optimizer = tf.train.GradientDescentOptimizer(0.0005)


# In[9]:


grads_and_vars = optimizer.compute_gradients(loss)


# In[10]:


grads, _ = list(zip(*grads_and_vars))
norms = tf.global_norm(grads)


# In[11]:


train_step = optimizer.minimize(loss)
#train_step = optimizer.minimize(norms)


# In[ ]:





# In[12]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[ ]:





# In[13]:


norms_ls=[] #store gradients norm per step
loss_ls=[] #store loss per step
epo_loss_ls=[]


# In[ ]:


epochs = 1000
batch_size = 500
batch_per_epoch = 100

for epo_num in range(epochs):
    rng_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)
    for i in range(batch_per_epoch):
        batch_xs, batch_ys = x_train[batch_size*i:batch_size*(i+1)].reshape((batch_size, 1)), y_train[batch_size*i:batch_size*(i+1)].reshape((batch_size, 1))
        _, norm, l= sess.run([train_step, norms, loss], feed_dict={x: batch_xs, y_: batch_ys})
        norms_ls.append(norm)
        loss_ls.append(l)

        if i % batch_per_epoch== 0:
            train_loss, train_norm = sess.run([loss, norms], feed_dict={x: x_train.reshape((-1,1)), y_: y_train.reshape((-1,1))})
            print("epoch: ", epo_num, ", loss: ", train_loss, ", norm: ", train_norm)
            epo_loss_ls.append(train_loss)



# In[ ]:


optimizer = tf.train.GradientDescentOptimizer(0.0000000001)
grads_and_vars = optimizer.compute_gradients(loss)
grads, _ = list(zip(*grads_and_vars))
norms = tf.global_norm(grads)
train_step2 = optimizer.minimize(norms)


# In[ ]:


epochs = 500
batch_size = 500
batch_per_epoch = 100

for epo_num in range(epochs):
    rng_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)
    for i in range(batch_per_epoch):
        batch_xs, batch_ys = x_train[batch_size*i:batch_size*(i+1)].reshape((batch_size, 1)), y_train[batch_size*i:batch_size*(i+1)].reshape((batch_size, 1))
        _, norm, l= sess.run([train_step2, norms, loss], feed_dict={x: batch_xs, y_: batch_ys})
        norms_ls.append(norm)
        loss_ls.append(l)

        if i % batch_per_epoch== 0:
            train_loss, train_norm = sess.run([loss, norms], feed_dict={x: x_train.reshape((-1,1)), y_: y_train.reshape((-1,1))})
            print("epoch: ", epo_num, ", loss: ", train_loss, ", norm: ", train_norm)
            epo_loss_ls.append(train_loss)




# In[ ]:





# In[ ]:


min(norms_ls)


# In[ ]:


norms_ls[-1]


# In[ ]:





# In[ ]:


#sample some point around the current point
stdr = 0.000003
test1 = tf.random_normal([1,40], 0, stddev=stdr)
W1s = W1+test1
test2 = tf.random_normal([40,40], 0, stddev=stdr)
W2s = W2+test2
test3 = tf.random_normal([40,1], 0, stddev=stdr)
W3s = W3+test3
sess.run(W1s)
sess.run(W2s)
sess.run(W3s)
print("done")


# In[ ]:



x2 = tf.placeholder("float", [None, 1]) #input
y2_ = tf.placeholder("float", [None, 1]) #ouput

XWb1 = tf.matmul(x2, W1s)+b1
outputs1 = tf.nn.relu(XWb1)

XWb2 = tf.matmul(outputs1, W2s)+b2
outputs2 = tf.nn.relu(XWb2)

XWb3 = tf.matmul(outputs2, W3s)+b3
output2 = XWb3

loss2 = tf.losses.mean_squared_error(labels=y2_, predictions=output2) # loss function


# In[ ]:






# In[ ]:


loss_around=[]
for i in range(100):
    lossv = sess.run(loss2,feed_dict={x2: x_train.reshape((-1,1)), y2_: y_train.reshape((-1,1))})
    loss_around.append(lossv)
    


# In[ ]:


loss_around = np.array(loss_around)


# In[ ]:

print("loss: ", epo_loss_ls[-1])
print("minimal ratio: ", np.sum(loss_around >= epo_loss_ls[-1])/100)










