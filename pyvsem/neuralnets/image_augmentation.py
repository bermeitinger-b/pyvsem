import numpy as np
from scipy import ndimage
from pyvsem.imageio import resize


def do_augmentation(images, max_shift, patch_shape, target_shape):
    max_x_shift, max_y_shift = max_shift
    patch_x, patch_y = patch_shape

    augmented_images = []

    for image in images:

        shift_x = np.random.randint(low=0, high=max_x_shift + 1)
        shift_y = np.random.randint(low=0, high=max_y_shift + 1)

        internal_shape = (image.shape[0], image.shape[1], image.shape[2])
        external_shape = (internal_shape[1], internal_shape[2], internal_shape[0])

        image = image.reshape(external_shape)[shift_x:shift_x + patch_x, shift_y:shift_y + patch_y]

        image = resize(image.reshape((internal_shape[0], patch_shape[0], patch_shape[1])), target_shape)

        if np.random.random() > 0.5:
            image = np.fliplr(image.reshape(external_shape))
            image = image.reshape(internal_shape)

        if np.random.random() > 0.5:
            image = np.flipud(image.reshape(external_shape))
            image = image.reshape(internal_shape)

        # four different possibilities to rotate the image (if quadratic)
        # two when not quadratic
        if image.shape[1] == image.shape[2]:
            rotate = np.random.random_integers(0, 4)
        else:
            rotate = np.random.random_integers(0, 2)

        # rotate == 0: don't rotate
        # rotate == 1: rotate 180 degree
        # rotate == 2: rotate  90 degree
        # rotate == 3: rotate 270 degree

        if rotate == 1:
            image = ndimage.rotate(image.reshape(external_shape), 180.0, (0, 1))
            image = image.reshape(internal_shape)
        elif rotate == 2:
            image = ndimage.rotate(image.reshape(external_shape), 90.0, (0, 1))
            image = image.reshape(internal_shape)
        elif rotate == 3:
            image = ndimage.rotate(image.reshape(external_shape), 270.0, (0, 1))
            image = image.reshape(internal_shape)

        # for CUDA, only single-precision works
        image = image.astype(np.float32)

        augmented_images.append(image)

    return np.asarray(augmented_images, dtype='float32')

if __name__ == '__main__':
    pass  # protection for joblib
