import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(1)
np.random.seed(1)
import input_data



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def save():
	print('This is save')
	x = tf.placeholder(tf.float32, [None, 784]) #input
	y_ = tf.placeholder(tf.float32, [None, 10]) #ouput

	h1 = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu, use_bias=False)
	h2 = tf.layers.dense(inputs=h1, units=512, activation=tf.nn.relu, use_bias=False)
	output = tf.layers.dense(inputs=h2, units=10, activation=None, use_bias=False)

	loss = tf.losses.softmax_cross_entropy(logits = output, onehot_labels= y_) # loss function
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train_step = optimizer.minimize(loss)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	sess = tf.Session() 
	saver = tf.train.Saver()

	#save init weights
	sess.run(tf.global_variables_initializer())
	saver.save(sess, './weight_init', write_meta_graph=False)

	for i in range(10000):
	    batch_xs, batch_ys = mnist.train.next_batch(100)
	    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	    if i % 500 == 0:
    		train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
    		val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
    		print("step ", i, ", loss: ", train_loss, ", acc: ", train_acc, ", val_loss: ", val_loss, ", val_acc: ", val_acc)

    #save final weights
	saver.save(sess, './weight_fin', write_meta_graph=False)  # meta_graph is not recommended
	print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	
	sess.close()


def restore():
	x = tf.placeholder(tf.float32, [None, 784]) #input
	y_ = tf.placeholder(tf.float32, [None, 10]) #ouput

	h1 = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu, use_bias=False)
	h2 = tf.layers.dense(inputs=h1, units=512, activation=tf.nn.relu, use_bias=False)
	output = tf.layers.dense(inputs=h2, units=10, activation=None, use_bias=False)

	loss = tf.losses.softmax_cross_entropy(logits = output, onehot_labels= y_) # loss function
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train_op = optimizer.minimize(loss)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
	sess = tf.Session() 
	saver = tf.train.Saver()

	# restore initial weights
	saver.restore(sess, './weight_init')
	info_array1 = sess.run(tf.trainable_variables())
	print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	
	# restore final weights
	saver.restore(sess, './weight_fin')
	info_array2 = sess.run(tf.trainable_variables())
	print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


	samples = np.linspace(0, 1., 2000)
	loss_list = []
	loss_list2 = []
	for i in samples:
		for idx, (j, z) in enumerate(zip(info_array1, info_array2)):
			sess.run(tf.assign(train_vars[idx], (1.-i)*j + i*z)) 
		train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
		val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
		loss_list.append(train_loss)
		loss_list2.append(val_loss)

	np.save('train_loss.npy', np.array(loss_list))
	np.save('val_loss.npy', np.array(loss_list2))
	sess.close()

# save()
print('--------------------')
tf.reset_default_graph()
restore()
