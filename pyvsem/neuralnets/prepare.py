import importlib
import os
import logging
import sklearn
import numpy as np
import argparse

from nolearn_utils.iterators import (
    ReadImageBatchIteratorMixin,
    BufferedBatchIteratorMixin,
    ShuffleBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    make_iterator
)
from pyvsem.neuralnets.options import AdjustVariable
import nolearn_utils.hooks

from pyvsem.utilities import save_pickle, load_pickle


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net",
        help="ID of the net, e.g. '26' or '19c'",
        type=str,
        required=True
    )
    parser.add_argument(
        "--colors",
        help="numbers of channels in the image: '3' for RGB or '1' for grey (default is %(default)s)",
        type=int,
        default=1
    )
    parser.add_argument(
        "--width",
        help="height and width to which the input images are rescaled",
        type=int,
        required=True
    )
    parser.add_argument(
        "--data-set",
        help="name of the data set on which to operate, must have a DataProvider and the images in the correct folder (e.g. 'mnist')",
        type=str,
        required=True
    )
    parser.add_argument(
        "--aug",
        help="flag to indicate if the input images should be augmented during training, defaults to %(default)s",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--task",
        help="Task to train: 'multi_class' or 'multi_label'",
        type=str,
        choices=['multi_class', 'multi_label'],
        required=True
    )
    parser.add_argument(
        "--batch-size",
        help="Number of images to process per batch: limited by GPU memory, defaults to %(default)s",
        type=int,
        default=128
    )
    return parser


def get_net_module(net_number):
    net = importlib.import_module('pyvsem.neuralnets.playground.net_{}'.format(net_number))
    return net


def get_data_provider(data_dir, data_set, task):

    if data_set is None:
        raise ValueError("No data set is given!")
    elif data_set.startswith('mirflickr-25k'):
        from pyvsem.neuralnets.datasets.Mirflickr25k import Mirflickr25kProvider as DataProvider
    elif data_set.startswith('neoclassica'):
        from pyvsem.neuralnets.datasets.Neoclassica import NeoclassicaProvider as DataProvider
    elif data_set.startswith('mnist'):
        from pyvsem.neuralnets.datasets.Mnist import MnistProvider as DataProvider
    elif data_set.startswith('imagenet'):
        from pyvsem.neuralnets.datasets.ImageNet import ImageNetProvider as DataProvider
    elif data_set.startswith('metmuseum'):
        from pyvsem.neuralnets.datasets.Metmuseum import MetmuseumProvider as DataProvider
    else:
        raise ValueError("This data set has no provider: '{}'".format(data_set))

    data_provider_pickle = os.path.join(data_dir, 'PROVIDER_{}.pickle'.format(task))

    if os.path.isfile(data_provider_pickle):
        data_provider = load_pickle(data_provider_pickle)
        log.info("Data provider loaded from pickle")
    else:
        data_provider = DataProvider(data_dir, task)
        save_pickle(data_provider, data_provider_pickle)
        log.info("Data provider stored to pickle")

    return data_provider


def get_data_dir(data_set):
    data_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir, os.pardir, 'data', data_set))
    if os.path.isdir(data_dir):
        log.info("Using '{}' as data directory".format(data_dir))
    else:
        raise ValueError("The given data directory does not exists: '{}'".format(data_dir))
    return data_dir


def get_data(data_dir, task, data_provider):
    x_train_pickle = os.path.join(data_dir, "XTRAIN_{}.pickle".format(task))
    x_test_pickle = os.path.join(data_dir, "XTEST_{}.pickle".format(task))
    y_train_pickle = os.path.join(data_dir, "YTRAIN_{}.pickle".format(task))
    y_test_pickle = os.path.join(data_dir, "YTEST_{}.pickle".format(task))

    if all([os.path.isfile(p) for p in [x_train_pickle, x_test_pickle, y_train_pickle, y_test_pickle]]):
        x_train = load_pickle(x_train_pickle)
        x_test = load_pickle(x_test_pickle)
        y_train = load_pickle(y_train_pickle)
        y_test = load_pickle(y_test_pickle)
        log.info("Train/test split loaded from pickles.")
    else:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            data_provider.get_image_paths(),
            data_provider.get_tags_for_images(),
            train_size=0.8)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        save_pickle(x_train, x_train_pickle)
        save_pickle(x_test, x_test_pickle)
        save_pickle(y_train, y_train_pickle)
        save_pickle(y_test, y_test_pickle)
        log.info("Train/test split stored to pickles")

    return x_train, x_test, y_train, y_test


def get_train_iterator(image_size, image_shape, batch_size, aug):
    train_iterator_mixins = [
        ReadImageBatchIteratorMixin,
        ShuffleBatchIteratorMixin
    ]

    if aug:
        train_iterator_mixins.extend([
            AffineTransformBatchIteratorMixin
        ])

    train_iterator_mixins.extend([
        BufferedBatchIteratorMixin,
    ])
    train_iterator = make_iterator('train_iterator', train_iterator_mixins)

    train_iterator_args = {
        'read_image_size': image_size,
        'read_image_as_gray': image_shape[0] == 1,
        'batch_size': batch_size,
        'buffer_size': 5,
        'verbose': 10
    }
    if aug:
        train_iterator_args.update({
            'affine_p': 0.5,
            'affine_scale_choices': np.linspace(0.75, 1.25, 5),
            'affine_translation_choices': np.arange(-5, 6, 1),
            'affine_rotation_choices': np.arange(-45, 50, 5),
        })
    train_iterator = train_iterator(**train_iterator_args)

    return train_iterator


def get_test_iterator(image_size, image_shape, batch_size):
    test_iterator_mixins = [
        ReadImageBatchIteratorMixin,
        BufferedBatchIteratorMixin
    ]
    test_iterator = make_iterator('test_iterator', test_iterator_mixins)

    test_iterator_args = {
        'read_image_size': image_size,
        'read_image_as_gray': image_shape[0] == 1,
        'batch_size': batch_size,
        'buffer_size': 5,
    }
    test_iterator = test_iterator(**test_iterator_args)

    return test_iterator


def get_on_epoch_finished(data_dir, current_run):
    return [
        nolearn_utils.hooks.PlotTrainingHistory(path=os.path.join(data_dir, "{}.pdf".format(current_run))),
        nolearn_utils.hooks.SaveTrainingHistory(path=os.path.join(data_dir, "{}_training_history.pickle".format(current_run))),
        nolearn_utils.hooks.EarlyStopping(patience=50),
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
    ]
