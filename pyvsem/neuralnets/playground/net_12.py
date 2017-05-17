import logging

import lasagne
import numpy as np
import theano
from nolearn.lasagne import NeuralNet

from pyvsem.neuralnets.options import AdjustVariable, EarlyStopping, sum_binary_crossentropy

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

            # input
            (lasagne.layers.InputLayer, {
                'name': 'input',
                'shape': (None, input_shape[0], input_shape[1], input_shape[2])
            }),

            # first conv
            (Conv2DLayer, {
                'name': 'conv_1',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_1',
                'pool_size': (2, 2),
                'stride': 2
            }),

            # dropout
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_1',
                'p': 0.5
            }),

            # one dense
            (lasagne.layers.DenseLayer, {
                'name': 'dense_1',
                'num_units': 1000
            }),

            # output
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

        batch_iterator_train=train_test_iterator,
        batch_iterator_test=train_test_iterator,

        objective_loss_function=sum_binary_crossentropy,

        #        custom_score=("validation score", custom_validation_score),

        max_epochs=max_epochs,

        verbose=2
    )
    return net
