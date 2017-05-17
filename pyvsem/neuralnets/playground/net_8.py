import logging

import lasagne
import numpy as np
import theano
from nolearn.lasagne import NeuralNet

from pyvsem.neuralnets.options import AdjustVariable, EarlyStopping, custom_validation_score, multi_label_log_loss

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

    # from:
    # "Very Deep ConvNets for Large-Scale Image Recognition"
    # by Karen Simonyan, Andrew Zisserman (Visual Geometry Group, University of Oxford)
    # http://www.image-net.org/challenges/LSVRC/2014/slides/VGG_ILSVRC_2014.pdf

    net = NeuralNet(
        layers=[

            # input
            (lasagne.layers.InputLayer, {
                'name': 'input',
                'shape': (None, input_shape[0], input_shape[1], input_shape[2])
            }),

            # first conv
            (Conv2DLayer, {
                'name': 'conv_1_1',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_1_2',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_1',
                'pool_size': (2, 2),
                'stride': 2
            }),

            # second conv
            (Conv2DLayer, {
                'name': 'conv_2_1',
                'num_filters': 64,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_2_2',
                'num_filters': 64,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_2',
                'pool_size': (2, 2),
                'stride': 2
            }),

            # third conv
            (Conv2DLayer, {
                'name': 'conv_3_1',
                'num_filters': 128,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_3_2',
                'num_filters': 128,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_3',
                'pool_size': (2, 2),
                'stride': 2
            }),

            # fourth conv
            (Conv2DLayer, {
                'name': 'conv_4_1',
                'num_filters': 256,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_4_2',
                'num_filters': 256,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_4',
                'pool_size': (2, 2),
                'stride': 2
            }),

            # fifth conv
            (Conv2DLayer, {
                'name': 'conv_5_1',
                'num_filters': 512,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_5_2',
                'num_filters': 512,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_5',
                'pool_size': (2, 2),
                'stride': 2
            }),

            # two fully connected
            (lasagne.layers.DenseLayer, {
                'name': 'dense_1',
                'num_units': 4096,
                # 'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (lasagne.layers.DenseLayer, {
                'name': 'dense_2',
                'num_units': 4096,
                # 'nonlinearity': lasagne.nonlinearities.rectify
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

        objective_loss_function=multi_label_log_loss,

        custom_score=("validation score", custom_validation_score),

        max_epochs=max_epochs,

        verbose=2
    )

    return net
