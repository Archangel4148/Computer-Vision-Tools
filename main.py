import numpy as np
from PIL import Image

from transformations import invert

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class SimpleImage:
    def __init__(self, width=50, height=50):
        # Create a blank image (black)
        self.width: int = width
        self.height: int = height
        self.data = np.zeros((height, width, 3), dtype=np.uint8)
        self.name: str | None = None

    def color_area(self, x_range, y_range, color, blend=False):
        """Set the area in the range to color (r, g, b)"""
        area = self.data[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        if blend:
            # Only overwrite channels that are non-zero in `color`
            mask = np.array(color) > 0
            for c in range(3):
                if mask[c]:
                    area[:, :, c] = color[c]
        else:
            area[:, :, :] = color

    def apply(self, func, **kwargs):
        """Apply a transformation to the image (pass in a function)"""
        self.data = func(self.data, **kwargs)

    def show(self, title=None):
        """Display the image"""
        plt.imshow(self.data)
        if title:
            plt.title(title)
        plt.show()

    def copy(self):
        new_img = SimpleImage(self.width, self.height)
        new_img.data = self.data.copy()
        return new_img

    @classmethod
    def from_file(cls, filepath):
        """Create a SimpleImage from an image file."""
        pil_img = Image.open(filepath).convert("RGB")  # ensure 3 channels
        data = np.array(pil_img, dtype=np.uint8)
        height, width = data.shape[:2]

        img = cls(width, height)
        img.data = data
        return img


import matplotlib.pyplot as plt


def show_images(images: list[SimpleImage], cols: int = None, images_per_row: int = 3, fig_size=(10, 5)):
    """Displays a list of images in a grid"""
    n = len(images)
    rows = (n - 1) // images_per_row + 1

    if cols is None:
        cols = n if n <= images_per_row else images_per_row

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = axes.flatten()  # Make it 1D so it's easy to loop through

    for i, img in enumerate(images):
        # Handle either SimpleImage or raw array
        data = img.data if isinstance(img, SimpleImage) else img
        axes[i].imshow(data)
        axes[i].axis('off')
        if img.name is not None:
            axes[i].set_title(img.name)

    # Turn off any extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


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

    # Load an image from a file
    flower_img = SimpleImage.from_file("images/flower.jpg")
    flower_img.name = "Flower Image"
    images_to_display.append(flower_img)

    # Invert the loaded image
    inverted_flower = flower_img.copy()
    inverted_flower.apply(invert)
    inverted_flower.name = "Inverted Flower Image"
    images_to_display.append(inverted_flower)

    # Display all the images
    show_images(images_to_display, images_per_row=2)


if __name__ == '__main__':
    main()
