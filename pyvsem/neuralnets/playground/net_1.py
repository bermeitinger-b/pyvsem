import logging

import theano
import lasagne
import numpy as np
from nolearn.lasagne import NeuralNet

from pyvsem.neuralnets.options import AdjustVariable, EarlyStopping, multi_label_log_loss

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s'
                    )
log = logging.getLogger(__file__)

# modules for Neural Nets CUDA
# import CUDA
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError as e:
    log.warning("CUDA not found, the fast layers are not used!")
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import MaxPool2DLayer


def create_net(input_shape, y, train_test_iterator, max_epochs=200, on_epoch_finished=list()):
    net = NeuralNet(
        layers=[
            (lasagne.layers.InputLayer, {
                'name': 'input',
                'shape': (None, input_shape[0], input_shape[1], input_shape[2])
            }),

            (Conv2DLayer, {
                'name': 'conv_1',
                'num_filters': 32,
                'filter_size': (3, 3)
            }),
            (MaxPool2DLayer, {
                'name': 'pool_1',
                'pool_size': (2, 2)
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_1',
                'p': 0.1
            }),

            (Conv2DLayer, {
                'name': 'conv_2',
                'num_filters': 64,
                'filter_size': (2, 2)
            }),
            (MaxPool2DLayer, {
                'name': 'pool_2',
                'pool_size': (2, 2)
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_2',
                'p': 0.2
            }),

            (Conv2DLayer, {
                'name': 'conv_3',
                'num_filters': 128,
                'filter_size': (2, 2)
            }),
            (MaxPool2DLayer, {
                'name': 'pool_3',
                'pool_size': (2, 2)
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_3',
                'p': 0.3
            }),

            (lasagne.layers.DenseLayer, {
                'name': 'hidden_4',
                'num_units': 1000
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_4',
                'p': 0.5
            }),

            (lasagne.layers.DenseLayer, {
                'name': 'hidden_5',
                'num_units': 1000
            }),

            (lasagne.layers.DenseLayer, {
                'name': 'output',
                'num_units': len(y),
                'nonlinearity': lasagne.nonlinearities.sigmoid
            })
        ],

        regression=True,

        update_learning_rate=theano.shared(np.cast['float32'](0.03)),
        update_momentum=theano.shared(np.cast['float32'](0.9)),

        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=max_epochs // 2)
        ],

        objective_loss_function=multi_label_log_loss,

        batch_iterator_train=train_test_iterator,
        batch_iterator_test=train_test_iterator,

        max_epochs=max_epochs,

        verbose=2
    )

    return net
