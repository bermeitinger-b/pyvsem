import os
import scipy.io
import logging

import numpy as np

from xml.etree import ElementTree

from pyvsem.neuralnets.datasets import DataProvider

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


class ImageNetProvider(DataProvider):
    def __init__(self, data_dir, task, min_amount=2, max_amount=np.inf):
        super().__init__(data_dir, task)

        log.debug("ImageNet-Provider using '{}' as working dir.".format(data_dir))

        image_dir = os.path.normpath(os.path.join(data_dir, 'images'))

        # find images
        class_paths = np.array(
            sorted(
                [os.path.join(image_dir, p) for p in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, p))]
            )
        )
        log.debug("Found {} classes".format(len(class_paths)))

        all_wnids = [os.path.basename(p) for p in class_paths]

        # convert wnids to actual words
        double_counter = 0
        wnid_to_def = {}
        tree = None
        with open(os.path.join(data_dir, 'structure_released.xml'), 'r') as f:
            tree = ElementTree.parse(f)
        root = tree.getroot()
        synsets = root[1]
        for child in synsets.iter():
            wnid = child.attrib.get('wnid')
            if wnid in all_wnids:
                definition = child.attrib.get('words')
                if definition in wnid_to_def.values():
                    wnid_to_def[wnid] = "{}_{}".format(definition, double_counter)
                    double_counter += 1
                else:
                    wnid_to_def[wnid] = definition

        log.debug("There are {} different synsets defined with the same definition.".format(double_counter))

        all_tags = []
        image_paths = []
        tags_for_images = []

        for p in class_paths:
            all_tags.append(wnid_to_def[os.path.basename(p)])
            for image in os.listdir(p):
                if image.endswith('.JPEG'):
                    if len(image_paths) % 100000 == 0:
                        log.debug("{:9} images processed".format(len(image_paths)))
                    image_paths.append(os.path.join(p, image))
                    tags_for_images.append(wnid_to_def[os.path.basename(p)])

        all_tags = np.array(all_tags)
        self.image_paths = np.array(image_paths)
        self.all_tags = self.fit_transform(all_tags)
        self.tags_for_images = np.array(self.transform(tags_for_images))

        log.debug("Number of images         : {:9}".format(len(self.image_paths)))
        log.debug("Number of tags for images: {:9}".format(len(self.tags_for_images)))
        log.debug("Number of all tags       : {:9}".format(len(self.all_tags)))
