import os
import time
import logging
import json
import pandas as pd
from collections import defaultdict, Counter

import numpy as np

from pyvsem.neuralnets.datasets import DataProvider

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(file)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


class MetmuseumProvider(DataProvider):
    def __init__(self, data_dir, task, min_amount=2, max_amount=np.inf):
        super().__init__(data_dir, task)

        log.debug("MetmuseumProvider using '{}' as working dir.".format(os.getcwd()))

        image_dir = os.path.normpath(os.path.join(data_dir, 'images'))
        class_dir = os.path.normpath(os.path.join(data_dir, 'annotations.csv'))

        # load annotations
        annotations = pd.read_csv(class_dir)
        log.debug("Found annotations for {} objects".format(len(annotations)))

        # load corresponding images
        images = annotations['Imagepath'].map(lambda x: os.path.abspath(x))

        log.debug("Found {} objects".format(len(images)))

        self.image_paths = annotations['Imagepath'].map(lambda x: os.path.join(data_dir, x))
        tags_for_images = [[t] for t in list(annotations['Classification'])]
        tag_dictionary = dict(Counter(t[0] for t in tags_for_images))

        self.image_paths = np.array(self.image_paths)

        log.info("Found {} images".format(len(self.image_paths)))

        images_without_tags = []
        log.info('Image count before filtering: {}'.format(len(tags_for_images) - len(images_without_tags)))

        started = time.time()
        ignored_tags = [tag for tag, count in tag_dictionary.items() if count <= min_amount or count >= max_amount]
        log.debug("Will ignore {} tags: {}".format(len(ignored_tags), ignored_tags))
        for i in range(len(tags_for_images)):
            if len(tags_for_images[i]) == 0:
                images_without_tags.append(i)
                continue  # empty tags, so don't remove anything
            tags_for_images[i] = np.array([k for k in tags_for_images[i] if k not in ignored_tags])
            if len(tags_for_images[i]) == 0:  # list may be empty now (=> no good tags)
                images_without_tags.append(i)
        log.info('Filtering tags finished in: {:.1f} s'.format(time.time() - started))
        log.info('Remaining images: {}'.format(len(tags_for_images) - len(images_without_tags)))

        tags_for_images = np.array(tags_for_images)

        self.image_paths = np.delete(self.image_paths, images_without_tags)
        self.tags_for_images = np.delete(tags_for_images, images_without_tags)

        if task == 'multi_class':
            log.info("Task is 'multi_class' so only take the first tag for each image.")
            # only use first tag
            self.tags_for_images = np.array([t[0] for t in self.tags_for_images])
            # no duplicates
            self.all_tags = np.array(list(set(self.tags_for_images)))
            # fit encoder
            self.fit(self.all_tags)
            # transform labels with encoder
            self.tags_for_images = [self.transform([t]) for t in self.tags_for_images]
            self.tags_for_images = np.array([i for j in self.tags_for_images for i in j])
        elif task == 'multi_label':
            log.info("Task is 'multi_label'")
            # use all tags, but no duplicates
            self.all_tags = np.array(list(set([k for ks in self.tags_for_images for k in ks])))
            # fit encoder
            self.fit(self.all_tags)
            # transform to vectors with encoder
            self.tags_for_images = np.array([self.transform(t)[0] for t in self.tags_for_images]).astype(np.float32)
            raise NotImplemented("This is not implemented")

        log.debug("Number of images         : {:9}".format(len(self.image_paths)))
        log.debug("Number of tags for images: {:9}".format(len(self.tags_for_images)))
        log.debug("Number of all tags       : {:9}".format(len(self.all_tags)))
