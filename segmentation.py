import tensorflow as tf
import numpy as np

def generalized_dice_loss(pred, true, eps=1E-64):
    """pred and true are tensors of shape (b, w_0, w_1, ..., c) where
             b   ... batch size
             w_k ... width of input in k-th dimension
             c   ... number of segments/classes
       furthermore, boths tensors have exclusively values in [0, 1]"""

    assert(pred.get_shape()[1:] == true.get_shape()[1:])

    shape_pred = pred.get_shape()
    shape_true = true.get_shape()
    prod_pred = reduce(lambda x,y:x*y, shape_pred[1:-1], tf.Dimension(1))
    prod_true = reduce(lambda x,y:x*y, shape_true[1:-1], tf.Dimension(1))

    # reshape to shape (b, W, c) where W is product of w_k
    pred = tf.reshape(pred, [-1, prod_pred, shape_pred[-1]])
    true = tf.reshape(true, [-1, prod_true, shape_true[-1]])

    # inverse square weighting for class cardinalities
    weights = tf.square(tf.reduce_sum(true, axis=[1]))+eps
    weights = tf.expand_dims(tf.reduce_sum(weights, axis=[-1]), -1)/weights

    # the traditional dice formula
    inter = tf.reduce_sum(weights*tf.reduce_sum(pred*true, axis=[1]), axis=[-1])
    union = tf.reduce_sum(weights*tf.reduce_sum(pred+true, axis=[1]), axis=[-1])

    return tf.reduce_mean(1.0-2.0*(inter+eps)/(union+eps))

def convert_to_mask(batch, threshold=0.5):

    result = np.zeros(batch.shape+(2,), dtype=batch.dtype)
    result[:,:,0] = batch >  threshold
    result[:,:,1] = batch <= threshold

    return result

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    batch_size, print_every, activation = 128, 1024, lambda x:0.5*(tf.tanh(x)+1)

    x  = tf.placeholder(tf.float32, [None, 784])
    x_ = tf.placeholder(tf.float32, [None, 784, 2])

    W  = tf.Variable(tf.zeros([784, 784, 2]))
    b  = tf.Variable(tf.zeros([784, 2]))

    y = activation(tf.tensordot(x, W, axes=[[1],[0]])+b)

    loss = generalized_dice_loss(y, x_)
    step = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for iteration in range(2**16):
        batch_x, _ = mnist.train.next_batch(batch_size)
        step_, loss_ = sess.run([step, loss],
                                feed_dict={x : batch_x,
                                           x_: convert_to_mask(batch_x)})

        if iteration % print_every == 0:
            print "loss :", loss_

    import matplotlib; matplotlib.use("Agg")
    import pylab as pl

    for index, image in enumerate(mnist.test.next_batch(batch_size)[0]):
        predict = sess.run(y, feed_dict={x: np.expand_dims(image, 0)})
        pl.subplot(131)
        pl.imshow(image.reshape((28, 28)))
        pl.subplot(132)
        pl.imshow(predict[0,:,0].reshape((28, 28)))
        pl.subplot(133)
        pl.imshow(predict[0,:,1].reshape((28, 28)))
        pl.savefig(str(index)+".png")
