import logging

import lasagne
import nolearn.lasagne
import theano
from numpy import cast

#from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def create_net(input_shape, y, train_iterator, test_iterator, max_epochs=250, on_epoch_finished=list()):

    # THIS IS "ConvNet A" VGG-16 with adjusting parameters (momentum, learning rate)

    net = nolearn.lasagne.NeuralNet(
        layers=[

            # input
            (lasagne.layers.InputLayer, {
                'name': 'input',
                'shape': (None, input_shape[0], input_shape[1], input_shape[2])
            }),

            # first conv
            (Conv2DLayer, {
                'name': 'conv1_1',
                'num_filters': 64,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool1',
                'pool_size': 2
            }),

            # second conv
            (Conv2DLayer, {
                'name': 'conv2_1',
                'num_filters': 128,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool2',
                'pool_size': 2
            }),

            # third conv
            (Conv2DLayer, {
                'name': 'conv3_1',
                'num_filters': 256,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv_2',
                'num_filters': 256,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool3',
                'pool_size': 2
            }),

            # fourth conv
            (Conv2DLayer, {
                'name': 'conv4_1',
                'num_filters': 512,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv4_2',
                'num_filters': 512,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool_4',
                'pool_size': 2
            }),


            # fifth conv
            (Conv2DLayer, {
                'name': 'conv5_1',
                'num_filters': 512,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (Conv2DLayer, {
                'name': 'conv5_2',
                'num_filters': 512,
                'filter_size': 3,
                'pad': 1,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (MaxPool2DLayer, {
                'name': 'pool5',
                'pool_size': 2
            }),

            # Fully connected 1
            (lasagne.layers.DenseLayer, {
                'name': 'fc6',
                'num_units': 4096,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout1',
                'p': 0.5
            }),

            # fully connected 2
            (lasagne.layers.DenseLayer, {
                'name': 'fc7',
                'num_units': 4096,
                'nonlinearity': lasagne.nonlinearities.rectify
            }),
            (lasagne.layers.DropoutLayer, {
                'name': 'dropout2',
                'p': 0.5
            }),


            # classification layer
            (lasagne.layers.DenseLayer, {
                'name': 'fc8',
                'num_units': len(y),
                'nonlinearity': lasagne.nonlinearities.softmax
            }),
        ],

        regression=False,

        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=theano.shared(cast['float32'](0.03)),
        update_momentum=theano.shared(cast['float32'](0.9)),

        on_epoch_finished=on_epoch_finished,

        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,

        objective_loss_function=lasagne.objectives.categorical_crossentropy,

        max_epochs=max_epochs,

        # added for compability,
        #train_split=nolearn.lasagne.TrainSplit(eval_size=0.1, stratify=False),

        verbose=10
    )
    net.initialize()
    return net
