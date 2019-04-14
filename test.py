import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def reshape_labels(labels):
    new_labels = []
    for y in labels:
        new_y = [0 for x in range(0,10)]
        new_y[y] = 1
        new_labels.append(new_y)
    return new_labels


digit_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()

train_labels = np.array(reshape_labels(train_labels))
test_labels = np.array(reshape_labels(test_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0
#print(test_images[5])

plt.figure()
plt.imshow(test_images[100])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = np.array([x.flatten() for x in train_images])
test_images = np.array([x.flatten() for x in test_images])

net = tf.Session()

# Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('temp/model.ckpt.meta')
saver.restore(net, "temp/model.ckpt")

#tf.reset_default_graph()
graph = tf.get_default_graph()
# Create some variables.
out = graph.get_tensor_by_name("out:0")
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")

feed_dict = {X:[test_images[100]]}
prediction = net.run(out, feed_dict)
print(prediction)
print(test_labels[100])