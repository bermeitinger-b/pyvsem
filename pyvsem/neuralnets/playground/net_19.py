import logging

import lasagne
import nolearn.lasagne

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def create_net(input_shape, y, train_test_iterator, max_epochs=200, on_epoch_finished=list()):

    net = nolearn.lasagne.NeuralNet(
        layers=[

            # input
            (lasagne.layers.InputLayer, {
                'name': 'input',
                'shape': (None, input_shape[0], input_shape[1], input_shape[2])
            }),

            # conv
            (lasagne.layers.Conv2DLayer, {
                'name': 'conv',
                'num_filters': 32,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (lasagne.layers.MaxPool2DLayer, {
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

        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=1e-2,
        update_momentum=0.9,

        batch_iterator_train=train_test_iterator,
        batch_iterator_test=train_test_iterator,

        on_epoch_finished=on_epoch_finished,

        objective_loss_function=lasagne.objectives.categorical_crossentropy,

        max_epochs=max_epochs,

        verbose=10
    )
    net.initialize()
    return net
