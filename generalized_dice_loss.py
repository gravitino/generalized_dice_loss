import tensorflow as tf

def generalized_dice_loss(pred, true, p=2, q=1, eps=1E-64):
    """pred and true are tensors of shape (b, w_0, w_1, ..., c) where
             b   ... batch size
             w_k ... width of input in k-th dimension
             c   ... number of segments/classes
       Furthermore, boths tensors have exclusively values in [0, 1].
       more than already good ones. The remaining parameters are as follows:
             p   ... power of inverse weigthing (p=2 default, p=0 uniform)
             q   ... power of inverse loss weighting (q=1 default, q=0 none)
             eps ... regularization term if empty classes occur"""

    assert(p   >= 0)
    assert(q   >= 0)
    assert(eps >= 0)
    assert(pred.get_shape()[1:] == true.get_shape()[1:])

    m = "the values in your last layer must be strictly in [0, 1]"
    with tf.control_dependencies([tf.assert_non_negative(pred, message=m),
                                  tf.assert_non_negative(true, message=m),
                                  tf.assert_less_equal(pred, 1.0, message=m),
                                  tf.assert_less_equal(true, 1.0, message=m)]):

        shape_pred = pred.get_shape()
        shape_true = true.get_shape()
        prod_pred = reduce(lambda x,y:x*y, shape_pred[1:-1], tf.Dimension(1))
        prod_true = reduce(lambda x,y:x*y, shape_true[1:-1], tf.Dimension(1))

        # reshape to shape (b, W, c) where W is product of w_k
        pred = tf.reshape(pred, [-1, prod_pred, shape_pred[-1]])
        true = tf.reshape(true, [-1, prod_true, shape_true[-1]])

        # inverse L_p weighting for class cardinalities
        weights = tf.abs(tf.reduce_sum(true, axis=[1]))**p+eps
        weights = tf.expand_dims(tf.reduce_sum(weights, axis=[-1]), -1)/weights

        # the traditional dice formula
        inter = tf.reduce_sum(weights*tf.reduce_sum(pred*true, axis=[1]),
                              axis=[-1])
        union = tf.reduce_sum(weights*tf.reduce_sum(pred+true, axis=[1]),
                              axis=[-1])

        loss = 1.0-2.0*(inter+eps)/(union+eps)

        if q == 0:
            return tf.reduce_mean(loss)

        # inverse L_q weighting for loss scores
        weights = tf.abs(loss)**q+eps
        weights = tf.reduce_sum(weights)/weights
        loss    = tf.reduce_sum(loss*weights)/tf.reduce_sum(weights)

        return loss

if __name__ == "__main__":
    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def convert_to_mask(batch, threshold=0.5):
        """toy model which segments image by thresholding"""

        result = np.zeros(batch.shape+(2,), dtype=batch.dtype)
        result[:,:,0] = batch >  threshold
        result[:,:,1] = batch <= threshold

        return result

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

    for iteration in range(2**14):
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
