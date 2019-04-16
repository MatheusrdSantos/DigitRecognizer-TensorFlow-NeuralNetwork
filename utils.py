import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from colr import color

def calc_accuracy(predictions, labels):
    total = len(predictions)
    count = 0
    for i in range(0, total):
        if(compare_predictions(predictions[i], labels[i])):
            count+=1
    return count/total
    
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

def getValue(matrix):
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

    feed_dict = {X:[matrix]}
    prediction = net.run(out, feed_dict)
    return prediction

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