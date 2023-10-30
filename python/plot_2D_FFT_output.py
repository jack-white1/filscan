import pandas as pd
import numpy as np
from PIL import Image

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

# Normalize log-transformed data to [0, 255] range and convert to uint8
normalized_data = ((log_image_data - log_image_data.min()) * (255.0 / (log_image_data.max() - log_image_data.min()))).astype(np.uint8)

# Convert numpy array to PIL Image and show
image = Image.fromarray(normalized_data)
image.show()

# save the image with the same name as the csv file but with .png extension
image.save(csv_file[:-4] + '.png')