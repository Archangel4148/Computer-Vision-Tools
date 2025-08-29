import cv2
import numpy as np


def invert(image: np.ndarray):
    # Instead of adding intensity to black pixels, subtract intensity from white pixels
    return 255 - image


def horizontal_flip(image: np.ndarray):
    # Same image, just iterate columns backwards
    return image[:, ::-1, :]


def vertical_flip(image: np.ndarray):
    # Same image, just iterate rows backwards
    return image[::-1, :, :]


def grayscale(image: np.ndarray):
    # Convert RGB values to raw intensity values (using luminosity constants from Wikipedia)
    grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return grayscale.astype(np.uint8)


def binarize(image: np.ndarray, threshold: tuple[int, int]):
    """Binarize an image to a given threshold"""
    min_val, max_val = threshold
    gray = grayscale(image)

    # Each pixel within the threshold is white, everything else is black
    binary = np.where((gray >= min_val) & (gray <= max_val), 255, 0)

    # Convert the 2D binary image (no color data) back into an RGB image (3 color channels)
    binary_rgb = np.stack([binary] * 3, axis=-1).astype(np.uint8)

    return binary_rgb


def adjust_brightness(image: np.ndarray, factor: float):
    """Multiply pixel brightness by factor"""
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def apply_kernel(img: np.ndarray, kernel: np.ndarray, convolve=True):
    """Convolve the image using the provided kernel"""
    # TODO: Use the faster version if possible (split the kernel into factors)
    # Pad the image based on the width of the kernel
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    padded_result = padded.copy()
    height, width, _ = padded.shape

    if convolve:
        # Flip the kernel for a convolution
        kernel = np.flip(kernel)

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


def apply_kernel_fast(img: np.ndarray, kernel: np.ndarray, convolve=True) -> np.ndarray:
    """Apply the kernel using OpenCV (probably a LOT faster)"""
    if convolve:
        kernel = np.flip(kernel)  # Flip the kernel for convolution
    return cv2.filter2D(img, -1, kernel)
