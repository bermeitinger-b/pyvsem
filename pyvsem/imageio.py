import os

import skimage.color
import skimage.io
import skimage.transform

import numpy as np
import skimage.color
import skimage.filters
import skimage.util
import skimage.util.dtype

# import joblib


def read_gray_image(path_to_image):
    if not os.path.isfile(path_to_image):
        raise ValueError('not a file: {}'.format(path_to_image))

    image = skimage.io.imread(path_to_image, as_grey=True)
    image = image.astype(np.float32)
    image /= 255.0

    # for internal reasons, we need 1xWxH
    image = image.reshape((1, image.shape[0], image.shape[1]))

    return image


def read_color_image(path_to_image):
    if not os.path.isfile(path_to_image):
        raise ValueError('not a file: {}'.format(path_to_image))

    image = skimage.io.imread(path_to_image)
    image = image.astype(np.float32)
    image /= 255.0

    # for internal reasons, we need a 3xWxH image instead of WxHx3
    image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))

    return image


def read_color_images(paths_to_images, rescale=None):
    # images = joblib.Parallel(n_jobs=joblib.cpu_count() // 2)(joblib.delayed(read_color_image)(img) for img in paths_to_images)
    images = [read_color_image(img) for img in paths_to_images]

    if rescale is not None:
        images = [resize(img, rescale) for img in images]

    return images


def read_gray_images(paths_to_images, rescale=None):
    # images = joblib.Parallel(n_jobs=joblib.cpu_count() // 2)(joblib.delayed(read_gray_image)(img) for img in paths_to_images)
    images = [read_gray_image(img) for img in paths_to_images]

    if rescale is not None and all((img.shape[1], img.shape[2]) != rescale for img in images):
        images = [resize(img, rescale) for img in images]

    return images


def resize(image, scale, mode=None):
    """
    Resizes the given image
    :param image:
    :param scale:
    :param mode: not used
    :return:
    """

    if not isinstance(scale, tuple):
        scale = (int(image.shape[0] * scale), int(image.shape[1] * scale))

    image = skimage.transform.resize(image.reshape((image.shape[1], image.shape[2], image.shape[0])), scale)
    image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))

    image = image.astype('float32')

    return image


def rgb2gray(image):
    return skimage.color.rgb2gray(image)


def rgb2hsv(image):
    return skimage.color.rgb2hsv(image)


def smooth(image, σ):
    return skimage.filters.gaussian_filter(image=image, sigma=σ).astype(dtype='float32')
