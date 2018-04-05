
# coding: utf-8

# # Set different batch size for representation of different model 

# In[166]:

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# B_SIZE = 5


# # load MNIST data

# In[167]:


import input_data


def model(B_SIZE):

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    l=[]
    acc =[]
    t_l = []
    t_acc = []
    norm = []    
    for j in B_SIZE:
        print('batch size now:', j)
        x = tf.placeholder("float", [None, 784]) #input
        y_ = tf.placeholder("float", [None, 10]) #ouput
    
        h1 = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu, use_bias=False)
        h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu, use_bias=False)
        #h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu, use_bias=False)        
        output = tf.layers.dense(inputs=h2, units=10, activation=None, use_bias=False) 
        loss = tf.losses.softmax_cross_entropy(logits = output, onehot_labels= y_) # loss fun ction
    
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)
    
        grads = tf.gradients(loss, x)
        norms = tf.global_norm(grads)
        
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        itr =30000
        # if j<10:
        #     itr = 10000
        # elif j<1000:
        #     itr = 8000            
        # elif j<10000:
        #     itr = 6000
        # else:
        #     itr = 4000
        for i in range(itr):
            batch_xs, batch_ys = mnist.train.next_batch(j)

            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 500 == 0:
                train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
                val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                print("epoch ", i/500, ", loss: ", train_loss, ", acc: ", train_acc, ", val_loss: ", val_loss, ", val_acc: ", val_acc)

        print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        
        images = np.vstack((mnist.train.images, mnist.validation.images))
        labels = np.vstack((mnist.train.labels, mnist.validation.labels))
        
        l.append(sess.run(loss, feed_dict={x: images, y_: labels})) #get training loss
        acc.append(sess.run(accuracy, feed_dict={x: images, y_: labels})) #get training acc.
        t_l.append(sess.run(loss, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #get testing loss
        t_acc.append(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #get testing acc.
        norm.append(sess.run(norms, feed_dict={x: images, y_: labels})) #get fro. norm of gradients
        
        sess.close()
        print('Fin. for this batch size!')
        print('___________________________________________')
        
    return np.array(l), np.array(acc), np.array(t_l), np.array(t_acc), np.array(norm)


b_size = [1,5,10,50,100,500,1000,5000,10000]
tr_l, tr_acc, te_l, te_acc, grad_n = model(b_size)
b_size = np.log10(np.array(b_size)) #log scale


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(b_size, np.array(tr_l), 'b')
ax1.plot(b_size, np.array(te_l), 'b',linestyle='--')
ax1.legend(['train loss','test loss'], loc = 'upper center')
ax1.set_ylabel('Loss', color="blue")
plt.xlabel('Batch size(scale: log)')

ax2 = ax1.twinx()  # this is the important function
ax2.plot(b_size, np.array(grad_n), 'r')
ax2.set_ylabel('Sensitivity', color="red")
ax2.legend(['Sensitivity'], loc = 'upper left')
plt.title('Loss, sensitivity v.s. batch size')
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(b_size, np.array(tr_acc), 'b')
ax1.plot(b_size, np.array(te_acc), 'b',linestyle='--')
ax1.legend(['train acc.','test acc.'], loc = 'upper center')
ax1.set_ylabel('Accuracy', color="blue")
plt.xlabel('Batch size(scale: log)')

ax2 = ax1.twinx()  # this is the important function
ax2.plot(b_size, np.array(grad_n), 'r')
ax2.set_ylabel('Sensitivity', color="red")
ax2.legend(['Sensitivity'], loc = 'upper left')
plt.title('Accuracy, sensitivity v.s. batch size')
plt.show()