import os
import time
import logging

import numpy as np

from pyvsem.neuralnets.datasets import DataProvider

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


class NeoclassicaProvider(DataProvider):
    def __init__(self, data_dir, task, min_amount=2, max_amount=np.inf):
        super().__init__(data_dir, task)

        log.debug("Neoclassica-Provider using '{}' as working dir.".format(os.getcwd()))

        image_dir = os.path.normpath(os.path.join(data_dir, 'images'))
        class_dir = os.path.normpath(os.path.join(data_dir, 'tags'))

        # find images
        self.image_paths = np.array(
            sorted(
                [os.path.join(image_dir, p) for p in os.listdir(image_dir) if p.endswith('.jpg')],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])))
        log.debug("Found {} images".format(len(self.image_paths)))

        # find tag files
        self.tag_paths = np.array(
            sorted(
                [os.path.join(class_dir, p) for p in os.listdir(class_dir) if p.endswith('.txt')],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])))
        log.debug("Found {} tag files".format(len(self.tag_paths)))

        tag_dictionary = {}
        images_without_tags = []
        tags_for_images = []

        started = time.time()

        for i in range(len(self.tag_paths)):
            with open(self.tag_paths[i], mode='r') as infile:
                tags = [tag.strip() for tag in infile.readlines()]
                tags_for_images.append(tags)
                for tag in tags:
                    if tag in tag_dictionary:
                        tag_dictionary[tag] += 1
                    else:
                        tag_dictionary[tag] = 1

        tags_for_images = np.array(tags_for_images)

        log.debug("Reading all tag files finished in: {:.1f} s".format(time.time() - started))

        log.info('Image count before filtering: {}'.format(len(tags_for_images) - len(images_without_tags)))

        started = time.time()
        ignored_tags = [tag for tag, count in tag_dictionary.items() if count <= min_amount or count >= max_amount]
        for i in range(len(tags_for_images)):
            if len(tags_for_images[i]) == 0:
                images_without_tags.append(i)
                continue  # empty tags, so don't remove anything
            tags_for_images[i] = [k for k in tags_for_images[i] if k not in ignored_tags]
            if len(tags_for_images[i]) == 0:  # list may be empty now (=> no good tags)
                images_without_tags.append(i)
        log.info('Filtering tags finished in: {:.1f} s'.format(time.time() - started))
        log.info('Remaining images: {}'.format(len(tags_for_images) - len(images_without_tags)))

        self.image_paths = np.delete(self.image_paths, images_without_tags)
        self.tags_for_images = np.delete(tags_for_images, images_without_tags)

        if task == 'multi_class':
            log.info("Task is 'multi_class' so only take the first tag for each image.")
            # only use first tag
            self.tags_for_images = [t[0] for t in self.tags_for_images]
            # no duplicates
            self.all_tags = np.array(list(set(self.tags_for_images)))
            # fit encoder
            self.fit(self.all_tags)
            # transform labels with encoder
            self.tags_for_images = np.array([self.transform(t) for t in self.tags_for_images])
        elif task == 'multi_label':
            log.info("Task is 'multi_label'")
            # use all tags, but no duplicates
            self.all_tags = np.array(list(set([k for ks in self.tags_for_images for k in ks])))
            # fit encoder
            self.fit(self.all_tags)
            # transform to vectors with encoder
            self.tags_for_images = np.array([self.transform(t)[0] for t in self.tags_for_images]).astype(np.float32)

        log.debug("Number of images         : {:9}".format(len(self.image_paths)))
        log.debug("Number of tags for images: {:9}".format(len(self.tags_for_images)))
        log.debug("Number of all tags       : {:9}".format(len(self.all_tags)))
