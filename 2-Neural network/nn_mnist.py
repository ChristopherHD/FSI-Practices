import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set[:7000]
train_y = one_hot(train_y, 10)

valid_x, valid_y = valid_set[7000:8500]
valid_y = one_hot(valid_y, 10)

test_x, test_y = test_set[8500:]
test_y = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# 784 entradas y 8 neuronas
W1 = tf.Variable(np.float32(np.random.rand(784, 8)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(8)) * 0.1)

# 8 entradas y 10 neuronas resultado
W2 = tf.Variable(np.float32(np.random.rand(8, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_size = 20
last_valid_data_error = 1000000
valid_data_error = 0
epoch = 0

while last_valid_data_error - valid_data_error >= 0.001:#for epoch in xrange(100):
    epoch += 1
    valid_mean_error = 0
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    if epoch > 1:
        last_valid_data_error = valid_data_error

    # Conjunto validacion
    valid_data_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    valid_mean_error += valid_data_error
    exitos_val=(len(valid_x)-valid_data_error)/len(valid_x)*100
    print "Epoch #:", epoch, "Error validacion: ", valid_data_error, "Exitos:", exitos_val
    print "Media de error de validacion en la epoca #:", epoch, ":", valid_mean_error / (
    (len(train_x) / batch_size)), ". Porcentaje de exito: ", exitos_val, "%"

    print "Epoch #:", epoch, "; Error Data: ", \
        valid_data_error, "; Error prev. data: ", last_valid_data_error, \
        "; Diferencia: ", last_valid_data_error - valid_data_error

# Conjunto test
test_data = sess.run(y, feed_dict={x: test_x})
error = 0.0
total = 0.0
for b, r in zip(test_y, test_data):
    total += 1
    if np.argmax(b) != np.argmax(r):
        error += 1
print "Conjunto Test"
print "Epoch #: Test",  "Error test: ", error / total * 100.0, "Porcentaje de exito: ", (1.0-error / total)*100.0, "% de 10000 muestras"