import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from colr import color
import random

def printMnist(image_vector, dimensions=[28,28], bg=False):
    for index, pixel in enumerate(image_vector, start=1):
        if bg:
            back = (pixel, pixel, pixel)
        else:
            back = (255-pixel, 255-pixel, 255-pixel)
        if(index%dimensions[0]==0):
            print(color(' ', fore=back, back=back))
        else:
            print(color(' ', fore=back, back=back), end="")
def reshape_labels(labels):
    new_labels = []
    for y in labels:
        new_y = [0 for x in range(0,10)]
        new_y[y] = 1
        new_labels.append(new_y)
    return new_labels
def compare_predictions(vector1, vector2):
    if(np.argmax(vector1) == np.argmax(vector2)):
        return True
    return False

def calc_accuracy(predictions, labels):
    total = len(predictions)
    count = 0
    for i in range(0, total):
        if(compare_predictions(predictions[i], labels[i])):
            count+=1
    return count/total

digit_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()

train_labels = np.array(reshape_labels(train_labels))
test_labels = np.array(reshape_labels(test_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.array([x.flatten() for x in train_images])
test_images = np.array([x.flatten() for x in test_images])
#printMnist(train_images[0].flatten())

""" plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show() """
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

# Output layer (transpose!)
out = tf.add(tf.matmul(hidden_1, W_out), bias_out, name="out")

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Fit neural net
batch_size = 100
mse_train = []
mse_test = []

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
        if np.mod(i, 39) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: train_images, Y: train_labels}))
            mse_test.append(net.run(mse, feed_dict={X: test_images, Y: test_labels}))
            print("Ephoc: ", e)
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            predictions = net.run(out, feed_dict={X: test_images})
            print('accuracy: ', calc_accuracy(predictions, test_labels))
            # Prediction
            #pred = net.run(out, feed_dict={X: test_images})
            save_path = saver.save(net, "temp/model.ckpt")
predictions = net.run(out, feed_dict={X: test_images})
calc_accuracy(predictions, test_labels)



           