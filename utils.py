import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
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