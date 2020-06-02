""" Loads an image

You shouldn't need to change any of this code! (hopefully)
"""

import numpy as np
from PIL import Image


def load_image(filename):
    """ Load image.

    Args:
        filename: A string. The path to the data file.

    Returns:
        X is a matrix of ints where each element is the pixel intensity with shape [num_rows_of_image, num_cols_of_image]
    """
    img = Image.open(filename).convert('L')
    X = np.asarray(img)
    return X
