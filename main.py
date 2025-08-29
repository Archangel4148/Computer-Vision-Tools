import numpy as np

from image_tools import SimpleImage, show_images
from transformations import invert, apply_kernel, apply_kernel_fast, binarize

USE_OPENCV = True


def main():
    images_to_display = []

    apply_kernel_method = apply_kernel_fast if USE_OPENCV else apply_kernel

    # Make an image with some colored blocks
    # img = SimpleImage()
    # img.color_area((20, 30), (20, 30), (255, 0, 0), blend=True)  # red
    # img.color_area((25, 33), (28, 40), (0, 255, 0), blend=True)  # green
    # img.color_area((8, 18), (8, 18), (0, 0, 255), blend=True)  # blue

    # Load an image from a file
    img = SimpleImage.from_file("images/josh_pic.png")

    img.name = "Original Image"
    images_to_display.append(img)

    # Create an inverted version of the image
    inverted_img = img.copy()
    inverted_img.apply(invert)
    inverted_img.name = "Inverted Image"
    images_to_display.append(inverted_img)

    # Do a box blur on the original
    binarized_img = img.copy()
    binarized_img.apply(binarize, threshold=(100, 255))
    binarized_img.name = "Binarize"
    images_to_display.append(binarized_img)

    # Do a Gaussian blur on the image
    gauss_img = img.copy()
    gauss_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]) / 16
    gauss_img.apply(apply_kernel_method, kernel=gauss_kernel)
    gauss_img.name = "Gaussian Blur"
    images_to_display.append(gauss_img)

    # Sharpen the image
    sharp_img = img.copy()
    sharp_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    sharp_img.apply(apply_kernel_method, kernel=sharp_kernel)

    sharp_img.name = "Sharpen Filter"
    images_to_display.append(sharp_img)

    # Emboss the image
    embossed_img = img.copy()
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2],
    ])
    embossed_img.apply(apply_kernel_method, kernel=emboss_kernel)
    embossed_img.name = "Emboss Filter"
    images_to_display.append(embossed_img)

    # Left-facing edge detection
    left_detected_img = img.copy()
    left_edge_kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ])
    left_detected_img.apply(apply_kernel_method, kernel=left_edge_kernel)
    left_detected_img.name = "Left Edge Detection"
    images_to_display.append(left_detected_img)

    # Edge detection (Gaussian blur, then binarize, then use edge detection kernel)
    edge_detected_img = img.copy()
    edge_detection_kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])
    edge_detected_img.apply(apply_kernel_method, kernel=gauss_kernel)
    edge_detected_img.apply(binarize, threshold=(100, 255))
    edge_detected_img.apply(apply_kernel_method, kernel=edge_detection_kernel)
    edge_detected_img.name = "Edge Detection"
    images_to_display.append(edge_detected_img)

    # Display the images
    show_images(images_to_display, images_per_row=4)


if __name__ == '__main__':
    main()
