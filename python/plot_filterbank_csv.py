import pandas as pd
import numpy as np
from PIL import Image

# Load the CSV data without headers
df = pd.read_csv('stretchedData.csv', header=None)

# Convert data to integer type
df = df.astype(int)

# Get the max values for rows and columns to define the size of the image
max_row = df[0].max()
max_col = df[1].max()

# Create an empty array filled with zeros
image_data = np.zeros((max_row+1, max_col+1))

# Using numpy advanced indexing to populate the image data
image_data[df[0].values, df[1].values] = df[2].values

# Convert numpy array to PIL Image and show
image = Image.fromarray(image_data)
image.show()
