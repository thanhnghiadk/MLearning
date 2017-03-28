from __future__ import print_function

__docformat__ = 'restructedtext en'

from load_data import load_data

import numpy as np
import timeit

import theano
import theano.tensor as T
import os
import sys

# setting environment variables
N_EPOCHS = 20000
LEARNING_RATE = 0.01
DATASET = 'mnist.pkl.gz'
BATCH_SIZE = 20
MINIMUM_COST = 0.0182
L1_REG = 0.0000
L2_REG = 0.0001


# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
def nn_softmax(x, w, b):
    return T.nnet.softmax(T.dot(x, w) + b)


def nn_tanh(x, w, b):
    return T.tanh(T.dot(x, w) + b)


# Define the cost function
def cost(p_y_given_x, y):
    return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])


# Define the output of neural network
def y_given_x(w1, b1, w2, b2, x):
    l1_value = nn_tanh(x, w1, b1)
    return nn_softmax(l1_value, w2, b2)


# get y predict value from given y
def y_pred(w1, b1, w2, b2, x):
    y_out = y_given_x(w1, b1, w2, b2, x)
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

# L1 Regularization
def L1(w1, w2):
    return abs(w1).sum() + abs(w2).sum()


# L2 Regularization
def L2(w1, w2):
    return (w1 ** 2).sum() + (w2 ** 2).sum()


# mlp network
def mlp_optimization_mnist():
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
    n_hiden = 500
    n_out = 10

    # hiden layer
    rng = np.random.RandomState(1234)
    W_values = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in + n_hiden)),
            high=np.sqrt(6. / (n_in + n_hiden)),
            size=(n_in, n_hiden)
        ),
        dtype=theano.config.floatX
    )

    W1 = theano.shared(
        value=W_values,
        name='W1',
        borrow=True
    )
    b_values = np.zeros((n_hiden,), dtype=theano.config.floatX)
    b1 = theano.shared(value=b_values, name='b1', borrow=True)

    # output layer
    W2 = theano.shared(
        value=np.zeros(
            (n_hiden, n_out),
            dtype=theano.config.floatX
        ),
        name='W2',
        borrow=True
    )

    b2 = theano.shared(
        value=np.zeros(
            (n_out,),
            dtype=theano.config.floatX
        ),
        name='b2',
        borrow=True
    )

    params = [W1, b1, W2, b2]

    # regularization
    l1 = L1(W1, W2)
    l2 = L2(W1, W2)

    # output
    p_y_given_x = y_given_x(W1, b1, W2, b2, x)
    y_pred_given_x = y_pred(W1, b1, W2, b2, x)

    cost_func = (cost(p_y_given_x, y) + L1_REG * l1 + L2_REG * l2)

    gparams = [T.grad(cost_func, param) for param in params]

    updates = [(param, param - LEARNING_RATE * gparam)
        for param, gparam in zip(params, gparams)]

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
    mlp_optimization_mnist()
