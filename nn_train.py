import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from colr import color
from utils import *

digit_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()

train_labels = np.array(reshape_labels(train_labels))
test_labels = np.array(reshape_labels(test_labels))
printMnist(test_images[0].flatten())

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.array([x.flatten() for x in train_images])
test_images = np.array([x.flatten() for x in test_images])


n_neurons_1 = 128
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name="X")
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Y")

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([28*28, n_neurons_1]))
bias_hidden_1 = tf.Variable(weight_initializer([n_neurons_1]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_1, 10]))
bias_out = tf.Variable(bias_initializer([10]))

#Hidden layers
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

# Output layer
out = tf.nn.softmax(tf.add(tf.matmul(hidden_1, W_out), bias_out), name="out")

# Cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(Y,out)

# Optimizer
opt = tf.train.AdamOptimizer().minimize(cross_entropy)

# Init
net.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Fit neural net
batch_size = 100
cross_train = []
cross_test = []

# Run
epochs = 10
for e in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_images = train_images[shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    # Minibatch training
    for i in range(0, len(train_labels) // batch_size):
        start = i * batch_size
        batch_x = train_images[start:start + batch_size]
        batch_y = train_labels[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
        
        # Show progress
        if np.mod(i, 60) == 0:
            # MSE train and test
            cross_train.append(net.run(cross_entropy, feed_dict={X: train_images, Y: train_labels}))
            cross_test.append(net.run(cross_entropy, feed_dict={X: test_images, Y: test_labels}))
            print("Ephoc: ", e)
            print('MSE Train: ', cross_train[-1])
            print('MSE Test: ', cross_test[-1])
            predictions = net.run(out, feed_dict={X: test_images})
            print('accuracy: ', calc_accuracy(predictions, test_labels))
            print(predictions[0])
            # Prediction
            #pred = net.run(out, feed_dict={X: test_images})
            #save_path = saver.save(net, "temp/model.ckpt")
'''
predictions = net.run(out, feed_dict={X: test_images})
calc_accuracy(predictions, test_labels)
print(predictions[0])
'''



           