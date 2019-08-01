import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# import mnist digits data set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 50

# training data placeholder variables
# input is 28 x 28 image, 1 x 784 vector
x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 784])
# tf.conv2d() and max_pool() only take 4D data, so we reshape x
# tf.reshape takes [i, j, k, l] where i is number if training samples, jxk is height x width, and l is num of channels
x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
# output data placeholder
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10])


def create_convolutional_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # tf.nn.conv_2d, [height, width, channels, filter depth]
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # weight and bias initialization
    W = tf.Variable(tf.random.truncated_normal(conv_filter_shape, stddev=0.03), name="{}_W".format(name))
    b = tf.Variable(tf.random.truncated_normal([num_filters], stddev=0.03), name="{}_b".format(name))

    # convolution operator, [1, x, y, 1] [x,y] stride
    out_layer = tf.nn.conv2d(input_data, W, [1, 1, 1, 1], padding="SAME")

    # adding bias
    out_layer += b

    # ReLU
    out_layer = tf.nn.relu(out_layer)

    # max pooling operator
    kernel_size = [1, pool_shape[0], pool_shape[1], 1]
    # 2x2 maxpool window size
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool2d(out_layer, ksize=kernel_size, strides=strides, padding="SAME")

    return out_layer


# architecture:
# Conv_layer1: 32 5x5 filters with 32 2x2 max pool
# Conv_layer2: 64 5x5 filters with 64 2x2 max pool
# Fully connected layer: 3136 node input layer, 1000 node hidden layer, 10 node output layer
conv_layer1 = create_convolutional_layer(input_data=x_reshaped, num_input_channels=1, num_filters=32,
                                         filter_shape=[5,5], pool_shape=[2, 2], name="conv_layer1")
conv_layer2 = create_convolutional_layer(input_data=conv_layer1, num_input_channels=32, num_filters=64,
                                         filter_shape=[5, 5], pool_shape=[2, 2], name="conv_layer2")

# final convolutional layer output are 7x7 feature maps with a depth of 64
# 7 x 7 x 64 = 3136 nodes for the fully connected layers input
# we have to flatten convolutional layer output into a 1x3136 vector
flat_output = tf.reshape(conv_layer2, [-1, 3136])

# Initialize weights and bias for fully connected layer
W1 = tf.Variable(tf.random.truncated_normal([3136, 1000], stddev=0.03), name="W1")
b1 = tf.Variable(tf.random.truncated_normal([1000], stddev=0.03), name="b1")
W2 = tf.Variable(tf.random.truncated_normal([1000, 10], stddev=0.03), name="W2")
b2 = tf.Variable(tf.random.truncated_normal([10], stddev=0.03), name="b2")

# activation functions
hidden_out = tf.add(tf.matmul(flat_output, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

output = tf.add(tf.matmul(hidden_out, W2), b2)
y_predicted = tf.nn.softmax(output)

# cost function
# softmax method takes weight-sum output from previous layer and y_actual of training data
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

# optimiser
optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_func)

# accuracy
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# initialize
init_op = tf.compat.v1.global_variables_initializer()

# cost list for plot
cost = []

# session
with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / BATCH_SIZE)

    for epoch in range(EPOCHS):
        avg_cost = 0

        for i in range(total_batch):
            batch_x, batch_y, = mnist.train.next_batch(batch_size=BATCH_SIZE)
            _, c = sess.run([optimiser, cost_func], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        cost.append(avg_cost / total_batch)

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch: {} || Cost: {:.3f} || Test accuracy: {:.3f}".format(epoch+1, avg_cost, test_acc))

    print("Final accuracy: {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))

plt.plot(cost)
plt.show()
