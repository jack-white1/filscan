import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

csv_file = 'stretchedDataFFT.csv'

# Load the CSV data without headers
df = pd.read_csv(csv_file, header=None)

# Convert row and column data to integer type, but keep intensity as float
df[0] = df[0].astype(int)
df[1] = df[1].astype(int)

# Get the max values for rows and columns to define the size of the image
max_row = df[0].max()
max_col = df[1].max()

# Create an empty array filled with zeros
image_data = np.zeros((max_row+1, max_col+1))

# Using numpy advanced indexing to populate the image data
image_data[df[0].values, df[1].values] = df[2].values

# Take the logarithm of the intensities (adding a small value to avoid log(0))
log_image_data = np.log(image_data + 1e-9)

x = np.linspace(0, max_row, max_row+1)
y = np.linspace(0, max_col, max_col+1)
x, y = np.meshgrid(x, y)


# Transpose the log_image_data
log_image_data_transposed = log_image_data.T
print("Shape of log_image_data_transposed:", log_image_data_transposed.shape)

# Create a 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, log_image_data_transposed, cmap='viridis', edgecolor='none')


# Set the labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
ax.set_title('3D Surface Plot of Intensity Data')

# Display the plot
plt.show()

# Save the plot if needed
# fig.savefig(csv_file[:-4] + '_3Dplot.png')
