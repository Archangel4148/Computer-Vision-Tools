import numpy as np
from PIL import Image


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

    def apply(self, funcs, **kwargs):
        """Apply one or more transformation functions to the image (every function is given kwargs)"""
        if callable(funcs):
            # Single function
            self.data = funcs(self.data, **kwargs)
        elif isinstance(funcs, (list, tuple)):
            # Apply each function in sequence
            for func in funcs:
                self.data = func(self.data, **kwargs)
        else:
            raise TypeError(f"Invalid transformation(s) for apply(): {funcs}")

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

    def save(self, save_path: str):
        """Save the image to a file"""
        img = Image.fromarray(self.data.astype(np.uint8), "RGB")
        img.save(save_path)

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
