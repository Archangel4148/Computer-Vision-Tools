import numpy as np

from image_tools import SimpleImage, show_images
from transformations import invert, horizontal_flip, apply_kernel

# Color constants
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def main():
    images_to_display = []

    # Make an image with some colored blocks
    img = SimpleImage()
    img.color_area((20, 30), (20, 30), RED, blend=True)  # red
    img.color_area((25, 33), (28, 40), GREEN, blend=True)  # green
    img.color_area((8, 18), (8, 18), BLUE, blend=True)  # blue
    img.name = "Original Image"
    images_to_display.append(img)

    # Create an inverted version of the image
    inverted_img = img.copy()
    inverted_img.apply(invert)
    inverted_img.name = "Inverted Image"
    images_to_display.append(inverted_img)

    # Do a box blur on the original
    transformed_img = img.copy()
    box_kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]) / 9
    transformed_img.apply(apply_kernel, kernel=box_kernel)
    transformed_img.name = "Box Blur"
    images_to_display.append(transformed_img)

    # Load an image from a file
    flower_img = SimpleImage.from_file("images/flower.jpg")
    flower_img.name = "Flower Image"
    images_to_display.append(flower_img)

    # Invert the loaded image
    inverted_flower = flower_img.copy()
    inverted_flower.apply((invert, horizontal_flip))
    inverted_flower.name = "Inverted/Flipped Flower Image"
    images_to_display.append(inverted_flower)

    # Do a Gaussian blur on the image
    gauss_flower = flower_img.copy()
    gauss_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]) / 16
    gauss_flower.apply(apply_kernel, kernel=gauss_kernel)
    gauss_flower.name = "Gaussian Blur"
    images_to_display.append(gauss_flower)

    # Display all the images
    show_images(images_to_display, images_per_row=3)


if __name__ == '__main__':
    main()
