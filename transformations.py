import numpy as np


# This file will contain all functions to perform transformations on an image

def invert(image: np.ndarray):
    return 255 - image


def horizontal_flip(image: np.ndarray):
    return image[:, ::-1, :]


def vertical_flip(image: np.ndarray):
    return image[::-1, :, :]


def apply_kernel(img: np.ndarray, kernel, convolve=True):
    # Pad the image based on the width of the kernel
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    padded_result = padded.copy()
    height, width, _ = padded.shape

    if convolve:
        # Flip the kernel for a convolution
        kernel = np.fliplr(np.flipud(kernel))

    # Loop the kernel through the whole image (skipping over padding)
    for y_p in range(pad_h, height - pad_h):
        for x_p in range(pad_w, width - pad_w):
            # Get the pixels in the kernel area
            kernel_area = padded[y_p - pad_h:y_p + pad_h + 1, x_p - pad_w:x_p + pad_w + 1, :]

            # Take the weighted sum of the kernel area
            for channel in range(3):
                padded_result[y_p, x_p, channel] = np.clip(np.sum(kernel_area[:, :, channel] * kernel), 0, 255).astype(
                    np.uint8)

    # Crop off the padding
    img[:, :, :] = padded_result[pad_h:-pad_h, pad_w:-pad_w, :]

    return img
