######################################################
# Multi-Layer Perceptron Classifier for MNIST dataset
# Mark Harvey
# Dec 2018
######################################################
import tensorflow as tf
import os
import sys
import shutil


# hyper-parameters
learning_rate = 0.001
batch_size = 50
steps = int(60000 / batch_size)


#####################################################
# Set up directories
#####################################################

# Returns the directory the current script (or interpreter) is running in
def get_script_directory():
    path = os.path.realpath(sys.argv[0])
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)


# create a directory for the MNIST dataset if it doesn't already exist
SCRIPT_DIR = get_script_directory()
MNIST_DIR = os.path.join(SCRIPT_DIR, 'mnist_dir')
if not (os.path.exists(MNIST_DIR)):
    os.makedirs(MNIST_DIR)
    print("Directory " , MNIST_DIR ,  "created ")

# create a directory for the TensorBoard data if it doesn't already exist
# delete it and recreate if it already exists
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_log')

if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 



#####################################################
# Dataset preparation
#####################################################
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)



#####################################################
# Create the Graph
# Define placeholders, the network, optimizer, 
# loss & accuracy nodes
#####################################################

# define placeholders for the input data & labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# The fully connected network model
#  - dense, fully-connected layer of 196 nodes, reLu activation
#  - dense, fully-connected layer of 10 nodes, softmax activation
input_layer = tf.layers.dense(inputs=x, kernel_initializer=tf.glorot_uniform_initializer(), units=196, activation=tf.nn.relu)
prediction = tf.layers.dense(inputs=input_layer, kernel_initializer=tf.glorot_uniform_initializer(), units=10, activation=tf.nn.softmax)


# Define a cross entropy loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=prediction, onehot_labels=y))

# Define the optimizer function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables
init = tf.initializers.global_variables()

# TensorBoard data collection
tf.summary.scalar('cross_entropy loss', loss)
tf.summary.scalar('accuracy', accuracy)


#####################################################
# Create & run the Session
#####################################################

# Launch the graph
with tf.Session() as sess:

    sess.run(init)

    # TensorBoard writer
    writer = tf.summary.FileWriter(TB_LOG_DIR, sess.graph)
    tb_summary = tf.summary.merge_all()

    # Training cycle
    for step in range(steps):
        batch = mnist.train.next_batch(batch_size)

        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})

        # display training accuracy every 100 batches
        if step % 100 == 0:
            print("Train step: {stp} -  Training accuracy: {acc}" .format(stp=step, acc=train_accuracy))

        # executing the optimizer is the actual training
        _, s = sess.run([optimizer,tb_summary], feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, step)

    print("Training Finished!")

    # Evaluation
    print ("Final Accuracy with test set:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


print ("FINISHED! Run tensorboard with: tensorboard --host localhost --port 6006 --logdir=./tb_log")
