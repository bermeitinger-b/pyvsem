import abc
import os
import numpy as np
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    LabelEncoder
)
from pyvsem.neuralnets.options import VALID_TASKS


class DataProvider(object):
    def __init__(self, data_dir, task):

        if not os.path.isdir(data_dir):
            raise ValueError("The data dir '{}' does not exist".format(data_dir))

        if task not in VALID_TASKS:
            raise ValueError("The task '{}' is not valid".format(task))

        self.data_dir = data_dir
        self.task = task

        if self.task == 'multi_class':
            self.encoder = LabelEncoder()
        elif self.task == 'multi_label':
            self.encoder = MultiLabelBinarizer()
        else:
            raise NotImplementedError()

        self.all_tags = []
        self.tags_for_images = []
        self.image_paths = []

    def get_image_paths(self):
        """
        Returns all paths for all images for the data set.
        """
        return np.array(self.image_paths)

    def get_tags_for_images(self):
        """
        Returns a list of lists where the n-th sublist contains
        all tags for the n-th image.
        """
        return np.array(self.tags_for_images)

    def get_all_tags(self):
        """
        Returns a set of unique tags in the data set.
        """
        return np.array(self.all_tags)

    def fit(self, y):
        if hasattr(self.encoder, 'classes_') and len(self.encoder.classes_) > 0:
            raise RuntimeError("DataProvider already fit")

        if isinstance(self.encoder, MultiLabelBinarizer):
            y = [y]

        self.encoder.fit(y)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def transform(self, y):
        if not hasattr(self.encoder, 'classes_') or len(self.encoder.classes_) < 1:
            raise RuntimeError("Encoder in DataProvider must be fit before transform.")

        if isinstance(self.encoder, MultiLabelBinarizer):
            return self.encoder.transform([y])[0].astype(np.float32)

        elif isinstance(self.encoder, LabelEncoder):
            return self.encoder.transform(y).astype(np.int32)

    def inverse_transform(self, yt):
        if isinstance(self.encoder, MultiLabelBinarizer):
            yt = np.around(yt).astype(np.int32)

        return self.encoder.inverse_transform(yt)


