import logging

import lasagne
import nolearn.lasagne

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

# modules for Neural Nets CUDA
# import CUDA
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
except ImportError as e:
    log.warning("CUDA not found, the fast layers are not used!")
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import MaxPool2DLayer


def create_net(input_shape, y, train_test_iterator, max_epochs=250):
    net = nolearn.lasagne.NeuralNet(
        layers=[

            # input
            (lasagne.layers.InputLayer, {
                'name': 'input',
                'shape': (None, input_shape[0], input_shape[1], input_shape[2])
            }),

            # first conv
            (Conv2DLayer, {
                'name': 'conv_11',
                'num_filters': 32,
                'filter_size': (3, 3),
                'pad': 'same',
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_12',
                'num_filters': 32,
                'filter_size': (3, 3),
                'pad': 'same',
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_1',
                'pool_size': 3,
                'stride': 2
            }),

            # second conv
            (Conv2DLayer, {
                'name': 'conv_21',
                'num_filters': 32,
                'filter_size': (3, 3),
                'pad': 'same',
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_22',
                'num_filters': 32,
                'filter_size': (3, 3),
                'pad': 'same',
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_2',
                'pool_size': 3,
                'stride': 2
            }),

            # dense+feature+dropout 1
            (lasagne.layers.DenseLayer, {
                'name': 'dense_1',
                'num_units': 256
            }),
            (lasagne.layers.FeaturePoolLayer, {
                'name': 'featurepool_1',
                'pool_size': 2
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_1',
                'p': 0.5
            }),
            
            # dense+feature+dropout 2
            (lasagne.layers.DenseLayer, {
                'name': 'dense_2',
                'num_units': 256
            }),
            (lasagne.layers.FeaturePoolLayer, {
                'name': 'featurepool_2',
                'pool_size': 2
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_2',
                'p': 0.5
            }),

            # output
            (lasagne.layers.DenseLayer, {
                'name': 'output',
                'num_units': len(y),
                'nonlinearity': lasagne.nonlinearities.softmax
            })
        ],

        regression=False,

        update=lasagne.updates.adam,
        update_learning_rate=1e-3,

        batch_iterator_train=train_test_iterator,
        batch_iterator_test=train_test_iterator,

        objective_loss_function=lasagne.objectives.categorical_crossentropy,

        max_epochs=max_epochs,

        verbose=10
    )
    net.initialize()
    return net

