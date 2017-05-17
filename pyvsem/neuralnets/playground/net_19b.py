import logging

import lasagne
import nolearn.lasagne
import theano
from numpy import cast

from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def create_net(input_shape, y, train_iterator, test_iterator, max_epochs=200, on_epoch_finished=list()):

    net = nolearn.lasagne.NeuralNet(
        layers=[

            # input
            (lasagne.layers.InputLayer, {
                'name': 'input',
                'shape': (None, input_shape[0], input_shape[1], input_shape[2])
            }),

            # conv
            (Conv2DLayer, {
                'name': 'conv',
                'num_filters': 32,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool',
                'pool_size': 2
            }),

            # dense
            (lasagne.layers.DenseLayer, {
                'name': 'dense',
                'num_units': 100,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),

            # output
            (lasagne.layers.DenseLayer, {
                'name': 'output',
                'num_units': len(y),
                'nonlinearity': lasagne.nonlinearities.softmax
            })
        ],

        regression=False,

        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=theano.shared(cast['float32'](0.03)),
        update_momentum=theano.shared(cast['float32'](0.9)),

        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,

        on_epoch_finished=on_epoch_finished,

        objective_loss_function=lasagne.objectives.categorical_crossentropy,

        max_epochs=max_epochs,

        verbose=10
    )
    return net
