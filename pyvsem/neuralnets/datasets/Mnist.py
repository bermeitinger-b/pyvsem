import os
import logging
import numpy as np

from pyvsem.neuralnets.datasets import DataProvider

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)


class MnistProvider(DataProvider):
    def __init__(self, data_dir, task):
        super().__init__(data_dir, task)

        log.debug("Mnist-Provider using '{}' as working dir.".format(os.getcwd()))

        image_dir = os.path.normpath(os.path.join(data_dir, 'images'))
        tag_file = os.path.normpath(os.path.join(data_dir, 'labels.txt'))

        # find images
        self.image_paths = np.array(
            sorted(
                [os.path.join(image_dir, p) for p in os.listdir(image_dir) if p.endswith('.jpg')],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])))
        log.debug("Found {} images".format(len(self.image_paths)))

        # read tag file
        tags_for_images = []

        with open(tag_file, mode='r') as infile:
            for tag in infile.readlines():
                tag = tag.rstrip()
                tags_for_images.append(tag)

        tags_for_images = np.array(tags_for_images)

        all_tags = np.array(list(set([k for ks in tags_for_images for k in ks])))
        self.all_tags = np.array(self.fit_transform(all_tags))
        self.tags_for_images = np.array([self.transform(x) for x in tags_for_images])

        log.debug("Number of images         : {:9}".format(len(self.image_paths)))
        log.debug("Number of tags for images: {:9}".format(len(self.tags_for_images)))
        log.debug("Number of all tags       : {:9}".format(len(self.all_tags)))
