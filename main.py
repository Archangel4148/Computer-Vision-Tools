import matplotlib.pyplot as plt
import numpy as np

# Create a 50x50 grid of black pixels
color_data = np.zeros((50, 50, 3), dtype=np.uint8)

# Draw some red pixels
color_data[20:30, 20:30, 0] = 255

# Draw some green pixels
color_data[28:40, 25:33, 1] = 255

# Draw some blue pixels
color_data[8:18, 8:18, 2] = 255

# Display the image using matplotlib
plt.imshow(color_data)
plt.show()
