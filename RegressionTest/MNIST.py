""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn



'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
#learning_rate = 0.001
learning_rate = 0.01
training_steps = 1000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)


## Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
    features = {"image_raw": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([], tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
    label = tf.cast(parsed_features['label'], tf.int32)
    return image, label

#def read_and_decode(filename_queue):
#    reader = tf.TFRecordReader()
#    _, serialized_example = reader.read(filename_queue)
#    features = tf.parse_single_example(
#        serialized_example,
#        # Defaults are not specified since both keys are required.
#        features={
#            'image_raw': tf.FixedLenFeature([], tf.string),
#            'label': tf.FixedLenFeature([], tf.int64),
#            'height': tf.FixedLenFeature([], tf.int64),
#            'width': tf.FixedLenFeature([], tf.int64),
#            'depth': tf.FixedLenFeature([], tf.int64)
#        })
#    image = tf.decode_raw(features['image_raw'], tf.uint8)
#    label = tf.cast(features['label'], tf.int32)
#    height = tf.cast(features['height'], tf.int32)
#    width = tf.cast(features['width'], tf.int32)
#    depth = tf.cast(features['depth'], tf.int32)
#    return image, label, height, width, depth

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
#training_filenames = ["/tmp/data/train.tfrecord"]
#filenames = tf.placeholder(tf.string, shape=[None])


filenames = ["/tmp/data/train.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(128)
iterator = dataset.make_one_shot_iterator()




# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}







def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def model_function(Data, Label):
    logits = RNN(Data, weights, biases)
    prediction = tf.nn.softmax(logits)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    return loss_op, prediction


def GetNext():
    batch_x, batch_y = iterator.get_next()
    batch_x = tf.reshape(batch_x,[batch_size, timesteps, num_input])
        
    batch_y = tf.one_hot(batch_y, num_classes)
    batch_y = tf.reshape(batch_y, tf.stack([batch_size, num_classes]))
    batch_y.set_shape([batch_size, num_classes])

    return batch_x, batch_y


loss_op, prediction = model_function(X, Y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    ## Restore variables from disk.
    #saver.restore(sess, "/tmp/model.ckpt")
    #print("Model restored.") 

    # Run the initializer
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for step in range(1, training_steps+1):
        
        ##Get Data From Demo Dataset from network
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        ## Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        batch_x, batch_y = sess.run(GetNext())

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))
    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")

    ## Save the variables to disk.
    #save_path = saver.save(sess, "/tmp/model.ckpt")
    #print("Model saved in file: %s" % save_path)

    ## Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = testdata.images[:test_len].reshape((-1, timesteps, num_input))
    #test_label = testdata.labels[:test_len]
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))