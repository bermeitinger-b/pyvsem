import logging

import lasagne
import nolearn.lasagne

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

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
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_12',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_13',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_1',
                'pool_size': (2, 2),
                'stride': 2
            }),

            (Conv2DLayer, {
                'name': 'conv_21',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_22',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_23',
                'num_filters': 32,
                'filter_size': (3, 3),
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_2',
                'pool_size': (2, 2),
                'stride': 2
            }),

            # dense 1
            (lasagne.layers.DenseLayer, {
                'name': 'dense_1',
                'num_units': 1500
            }),

            # dropout
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout_1',
                'p': 0.5
            }),

            # dense 2
            (lasagne.layers.DenseLayer, {
                'name': 'dense_2',
                'num_units': 1500
            }),

            # dropout
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

        update=lasagne.updates.adam,

        update_learning_rate=0.002,

        batch_iterator_train=train_test_iterator,
        batch_iterator_test=train_test_iterator,

        objective=regularization_objective,
        objective_lambda2=0.0025,

        max_epochs=max_epochs,

        verbose=3
    )
    net.initialize()
    return net


def regularization_objective(layers, lambda1=0.0, lambda2=0.0, *args, **kwargs):
    # default loss
    losses = nolearn.lasagne.objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = lasagne.layers.get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses
