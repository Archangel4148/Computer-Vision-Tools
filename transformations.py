import numpy as np


# This file will contain all functions to perform transformations on an image

def invert(image: np.ndarray):
    return 255 - image


def horizontal_flip(image: np.ndarray):
    return image[:, ::-1, :]


def vertical_flip(image: np.ndarray):
    return image[::-1, :, :]
