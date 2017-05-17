import joblib
import re
import os
import xml.dom.minidom
import random

import numpy as np

import sys
sys.setrecursionlimit(10000)


def read_dataset(params):
    filemask = r'.*(jpg|JPEG|gif)'

    image_paths, annotations, concept_list = [], [], []

    if params['input_format'] == 'complete_annotation':
        image_paths = sorted([f for f in os.listdir(params['image_dir']) if re.match(filemask, f)])
        annotations = []
        concept_list = []

        for image in image_paths:
            filename = os.path.splitext(image)[0]
            annotation_file = read_xml(os.path.join(params['annotations'], filename + '.xml'))
            object_names = annotation_file['object_names']
            annots = []
            for i in range(len(object_names)):
                if object_names[i] not in concept_list:
                    concept_list.append(object_names[i])
                annots.append(((annotation_file['boxes'][i]), object_names[i]))
            annotations.append(annots)

        concept_list = sorted(concept_list)

    elif params['input_format'] == 'concept_file':
        pass
    elif params['input_format'] == 'concept_folder':
        pass
    elif params['input_format'] == 'concept_folder_with_repetition':
        pass
    elif params['input_format'] == 'image_file':
        pass
    elif params['input_format'] == 'desc_files':
        pass

    # add full path
    image_paths = [os.path.join(params['image_dir'], f) for f in image_paths]

    return np.asarray(image_paths), np.asarray(annotations), np.asarray(concept_list)


# https://github.com/rbgirshick/fast-rcnn/blob/b0758d0a67f45bba9fbe64dca3b20b3a510bb389/lib/datasets/pascal_voc.py
def read_xml(xml_file):
    with open(xml_file, mode='r') as f:
        data = xml.dom.minidom.parseString(f.read())

    objects = data.getElementsByTagName('object')
    number_of_objects = len(objects)

    boxes = np.zeros((number_of_objects, 4), dtype=np.uint16)
    object_names = []

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    for ix, obj in enumerate(objects):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin'))
        y1 = float(get_data_from_tag(obj, 'ymin'))
        x2 = float(get_data_from_tag(obj, 'xmax'))
        y2 = float(get_data_from_tag(obj, 'ymax'))
        boxes[ix, :] = [x1, y1, x2, y2]
        object_names.append(get_data_from_tag(obj, 'name'))

    return {
        'boxes': boxes,
        'object_names': object_names,
    }


def random_dataset_subset(image_limit, images, annotations, random_seed=None):
    assert len(images) == len(annotations)
    assert len(images) >= image_limit
    indexes = get_random_indexes(images, image_limit, random_seed)
    return images[indexes], annotations[indexes]


def get_random_indexes(population, num_samples, random_seed=None):
    if random_seed is None:
        random.seed(random_seed)
    if isinstance(population, int):
        return random.sample(range(population), num_samples, )
    else:
        return random.sample(range(len(population)), num_samples)


class Options(object):
    def update(self, new_options):
        if isinstance(new_options, Options):
            self.update(new_options.get_all())
        else:
            if len(new_options) > 0:
                for key, value in new_options.items():
                    try:
                        self.__setattr__(key, value)
                    except AttributeError:
                        # no duck-punching
                        pass

    def get_all(self):
        return self.__dict__


def float_range(start, stop, step):
    return np.arange(start, stop + step, step)


def load_pickle(filename):
    return joblib.load(filename)


def save_pickle(object, filename):
    joblib.dump(object, filename)
