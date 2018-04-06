import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(1)
np.random.seed(1)
import input_data



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def save(state):
	print('This is save')
	x = tf.placeholder(tf.float32, [None, 784]) #input
	y_ = tf.placeholder(tf.float32, [None, 10]) #ouput

	h1 = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu, use_bias=False)
	h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu, use_bias=False)
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
# 	saver.save(sess, './weight_init', write_meta_graph=False)
	
	for i in range(200000):
		if(state == 1):
		    batch_xs, batch_ys = mnist.train.next_batch(64)
		else:
			batch_xs, batch_ys = mnist.train.next_batch(1024)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		if i % 500 == 0:
			train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
			val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			print("step ", i, ", loss: ", train_loss, ", acc: ", train_acc, ", val_loss: ", val_loss, ", val_acc: ", val_acc)

	#save final weights
	if(state == 1):
		saver.save(sess, './weight_model_1', write_meta_graph=False)  # meta_graph is not recommended
	else:
		saver.save(sess, './weight_model_2', write_meta_graph=False)  # meta_graph is not recommended
	print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	
	sess.close()


def restore():
	x = tf.placeholder(tf.float32, [None, 784]) #input
	y_ = tf.placeholder(tf.float32, [None, 10]) #ouput

	h1 = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu, use_bias=False)
	h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu, use_bias=False)
	output = tf.layers.dense(inputs=h2, units=10, activation=None, use_bias=False)

	loss = tf.losses.softmax_cross_entropy(logits = output, onehot_labels= y_) # loss function
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train_op = optimizer.minimize(loss)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
	sess = tf.Session() 
	saver = tf.train.Saver()

	# restore initial weights
	saver.restore(sess, './weight_model_1')
	info_array1 = sess.run(tf.trainable_variables())
	print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	
	# restore final weights
	saver.restore(sess, './weight_model_2')
	info_array2 = sess.run(tf.trainable_variables())
	print ("Accuracy(test set): ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


	samples = np.linspace(-1., 2., 10000)
	loss_list = []
	loss_list2 = []
	acc_list = []
	acc_list2 = []
	for i in samples:
		for idx, (j, z) in enumerate(zip(info_array1, info_array2)):
			sess.run(tf.assign(train_vars[idx], (1.-i)*j + i*z)) 
		train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
		val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
		loss_list.append(train_loss)
		loss_list2.append(val_loss)
		acc_list.append(train_acc)
		acc_list2.append(val_acc)

	np.save('./interpolation/train_loss_1.npy', np.array(loss_list))
	np.save('./interpolation/val_loss_1.npy', np.array(loss_list2))
	np.save('./interpolation/train_acc_1.npy', np.array(acc_list))
	np.save('./interpolation/val_acc_1.npy', np.array(acc_list2))
	sess.close()

save(1)
save(2)
print('--------------------')
tf.reset_default_graph()
restore()
