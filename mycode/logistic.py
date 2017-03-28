from __future__ import print_function

__docformat__ = 'restructedtext en'

from load_data import load_data

import numpy as np
import timeit

import theano
import theano.tensor as T
import os
import sys

# setting variables
N_EPOCHS = 20000
LEARNING_RATE = 0.13
DATASET = 'mnist.pkl.gz'
BATCH_SIZE = 600
MINIMUM_COST = 0.075


# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
def nn(x, w, b):
    return T.nnet.softmax(T.dot(x, w) + b)


# Define the cost function
def cost(p_y_given_x, y):
    return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])


def y_given_x(w, b, x):
    return nn(x, w, b)


def y_pred(w, b, x):
    y_out = nn(x, w, b)
    return T.argmax(y_out, axis=1)


def errors(y_pred, y):
    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', y_pred.type)
        )
    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(y_pred, y))
    else:
        raise NotImplementedError()


def lg_optimization_mnist():
    datasets = load_data(DATASET)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print('... data loaded')
    print('... building the model')

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')

    n_in = 28 * 28
    n_out = 10

    W = theano.shared(
        value=np.zeros(
            (n_in, n_out),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

    b = theano.shared(
        value=np.zeros(
            (n_out,),
            dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
    )

    p_y_given_x = y_given_x(W, b, x)
    y_pred_given_x = y_pred(W, b, x)

    cost_func = cost(p_y_given_x, y)

    g_W = T.grad(cost=cost_func, wrt=W)
    g_b = T.grad(cost=cost_func, wrt=b)

    updates = [(W, W - LEARNING_RATE * g_W),
               (b, b - LEARNING_RATE * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost_func,
        updates=updates,
        givens={
            x: train_set_x[index * BATCH_SIZE: (index + 1) * BATCH_SIZE],
            y: train_set_y[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
        }
    )

    valid_errors = errors(y_pred_given_x, y)

    validate_model = theano.function(
        inputs=[index],
        outputs=valid_errors,
        givens={
            x: valid_set_x[index * BATCH_SIZE: (index + 1) * BATCH_SIZE],
            y: valid_set_y[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
        }
    )

    print('... training the model')

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // BATCH_SIZE
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // BATCH_SIZE
    patience = 10000  # look as this many examples regardless
    validation_frequency = min(n_train_batches, patience // 2)

    done_looping = False
    epoch = 0
    best_validation_loss = np.inf
    start_time = timeit.default_timer()
    while (epoch < N_EPOCHS) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                if(this_validation_loss < best_validation_loss):
                    best_validation_loss = this_validation_loss
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                if this_validation_loss < MINIMUM_COST:
                    done_looping = True

    end_time = timeit.default_timer()
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))

if __name__ == '__main__':
    lg_optimization_mnist()
