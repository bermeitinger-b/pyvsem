import logging

import numpy as np
import sklearn.preprocessing
import theano
import theano.tensor as T
from nolearn.lasagne import BatchIterator

from pyvsem.imageio import read_color_images, read_gray_images
from pyvsem.neuralnets.image_augmentation import do_augmentation

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

VALID_TASKS = ['multi_label', 'multi_class']


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class ImageIterator(BatchIterator):
    def __init__(self, all_classes, image_shape, task, aug=True, batch_size=128):
        super().__init__(batch_size)

        if task not in VALID_TASKS:
            raise ValueError("The task '{}' is not valid.".format(task))

        if task == 'multi_label':
            self.binarizer = sklearn.preprocessing.MultiLabelBinarizer(classes=all_classes)
        elif task == 'multi_class':
            self.binarizer = sklearn.preprocessing.LabelEncoder()

        self.binarizer.fit(all_classes)

        self.__task = task
        self.__aug = aug

        self.__image_dimensions = image_shape[0]
        self.__image_shape_x = image_shape[1]
        self.__image_shape_y = image_shape[2]
        self.__patch_size_x = int(self.__image_shape_x * 0.8)
        self.__patch_size_y = int(self.__image_shape_y * 0.8)
        self.__max_x_shift = self.__image_shape_x - self.__patch_size_x
        self.__max_y_shift = self.__image_shape_y - self.__patch_size_y

    def transform(self, Xb, yb=None):

        if self.__image_dimensions == 1:  # grayscale
            images = read_gray_images(Xb, (self.__image_shape_x, self.__image_shape_y))
        else:
            images = read_color_images(Xb, (self.__image_shape_x, self.__image_shape_y))

        if yb is None:
            classes = None
        else:
            classes = self.binarizer.transform(yb)
            # use int32 because of theano
            classes = np.array(classes).astype(np.int32)

        # now for some augmentation:
        if self.__aug:
            images = do_augmentation(
                images,
                (self.__max_x_shift, self.__max_y_shift),
                (self.__patch_size_x, self.__patch_size_y),
                (self.__image_shape_x, self.__image_shape_y)
            )

        return images, classes


ONE = theano.shared(np.float32(1.0))
EPSILON = theano.shared(np.float32(1.0e-7))


def multi_label_log_loss(predictions, targets):
    """
    :param predictions: list of predictions in [0.0, 1.0]
    :param targets: list of target values in {0, 1}
    :return:
    """
    predictions = T.clip(predictions, EPSILON, ONE - EPSILON)

    return - T.sum(T.log(ONE - abs(targets - predictions))) / targets.shape[0]


def multi_label_cross_entropy(predictions, targets):
    one = theano.shared(np.float32(1.0))
    epsilon = theano.shared(np.float32(1.0e-7))
    predictions = T.clip(predictions, epsilon, one - epsilon)

    return - T.sum((targets * T.log(predictions)) + (one - targets) * T.log(one - predictions))


def custom_validation_score(x, y):
    return np.mean(np.abs(x - y))
