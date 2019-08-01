import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyperparameters
LEARNING_RATE = 0.5
EPOCHS = 10
BATCH_SIZE = 100

# declaring data variables
# shape is ? x 784 since we have an unspecified number of samples
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
# shape is ? x 10 since we have an unspecified number of samples
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# 3 layer neural net: [784, 300, 10]
# W1 dimensions: 784 x 300
# (1 x 784)(784 x 300) = (1 x 300)
W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name="W1")
b1 = tf.Variable(tf.random.normal([300], stddev=0.03), name="b1")

W2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name="W2")
b2 = tf.Variable(tf.random.normal([10], stddev=0.03), name="b2")

# operations
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.sigmoid(hidden_out)

output = tf.add(tf.matmul(hidden_out, W2), b2)
y_predicted = tf.nn.sigmoid(output)

# cost function
# tf.reduce_mean is needed
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

# optimiser
optimiser = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost_func)

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
    total_batch = int(len(mnist.train.labels)/BATCH_SIZE)

    for epoch in range(EPOCHS):
        avg_cost = 0

        for i in range(total_batch):
            batch_x, batch_y, = mnist.train.next_batch(batch_size=BATCH_SIZE)
            _, c = sess.run([optimiser, cost_func], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c/total_batch

        cost.append(avg_cost/total_batch)

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch: {} || Cost: {:.3f} || Test accuracy: {:.3f}".format(epoch+1, avg_cost, test_acc))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

plt.plot(cost)
plt.show()
