import os
import collections
import time
import logging

import numpy as np

from pyvsem.neuralnets.datasets import DataProvider

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)


class Mirflickr25kProvider(DataProvider):
    def __init__(self, data_dir, task, *args, **kwargs):
        super().__init__(data_dir, task)

        log.debug("Mirflickr25k-Provider using '{}' as working dir.".format(data_dir))

        self.data_dir = data_dir

        image_dir = os.path.normpath(os.path.join(data_dir, 'images'))

        # find images
        self.image_paths = np.array(
            sorted(
                [os.path.join(image_dir, p) for p in os.listdir(image_dir) if p.endswith('.jpg')],
                key=lambda x: int(x[len(image_dir) + 3:-4])))
        log.debug("Found {} images".format(len(self.image_paths)))

        if task == 'multi_label':
            self.min_amount = 20
            self.max_amount = 200
            self.__do_multi_label()
        else:
            self.__do_multi_class()

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

    def __do_multi_label(self):

        class_dir = os.path.normpath(os.path.join(self.data_dir, 'tags'))

        # find tag files
        self.tag_paths = np.array(
            sorted(
                [os.path.join(class_dir, p) for p in os.listdir(class_dir) if p.endswith('.txt')],
                key=lambda x: int(x[len(class_dir) + 5:-4])))
        log.debug("Found {} tag files".format(len(self.tag_paths)))

        tag_dictionary = {}
        images_without_tags = []
        tags_for_images = []

        started = time.time()

        # only do reading tag files and filtering if they are not already loaded

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
        ignored_tags = [tag for tag, count in tag_dictionary.items() if count <= self.min_amount or count >= self.max_amount]
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
        self.all_tags = np.array(list(set([k for ks in self.tags_for_images for k in ks])))

    def __do_multi_class(self):

        annotation_dir = os.path.join(self.data_dir, 'annotations')

        classes = collections.defaultdict(set)
        for anno_file in os.listdir(annotation_dir):
            anno_file = os.path.join(annotation_dir, anno_file)
            concept = os.path.splitext(os.path.basename(anno_file))[0]
            if not concept.endswith('_r1'):   # only use relevant
                continue
            with open(anno_file, 'r') as infile:
                for im in infile.readlines():
                    classes[concept].add(im.strip())

        log.debug("found {} relevant classes".format(len(classes)))

        duplicates = collections.defaultdict(set)
        for key, values in classes.items():
            for value in values:
                for key2, values2 in classes.items():
                    if key == key2:
                        continue
                    if value in values2:
                        duplicates[int(value)].add(key)
                        duplicates[int(value)].add(key2)
        log.debug("found {} duplicate entries".format(len(duplicates)))

        assigned_images = []
        for x in classes.values():
            assigned_images.extend(x)
        assigned_images = [int(x) for x in assigned_images]
        log.debug("{} images have classes assigned to them".format(len(assigned_images)))

        filtered = [i for i in self.image_paths
                    if int(os.path.basename(i)[2:-4]) not in duplicates and
                    int(os.path.basename(i)[2:-4]) in assigned_images]
        log.debug("filtered to {} non-duplicate relevant images".format(len(filtered)))

        self.image_paths = np.array(filtered)
        self.all_tags = np.array([x for x in classes.keys()])

        tags_for_images = []
        for im in filtered:
            im = os.path.splitext(os.path.basename(im))[0][2:]
            for concept, images in classes.items():
                if im in images:
                    tags_for_images.append(concept)
                    break

        self.tags_for_images = np.array(tags_for_images)

    def get_all_tags(self):
        return self.all_tags

    def get_tags_for_images(self):
        return self.tags_for_images

    def get_image_paths(self):
        return self.image_paths
