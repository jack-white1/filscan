import matplotlib.pyplot as plt
import numpy as np
import csv

# Read in the data from stretchedPaddedData.csv in format frequency bin, time bin, intensity
data = np.genfromtxt('stretchedPaddedData.csv', delimiter=',')

#plot the data as an image, save to png
plt.imshow(data, aspect='auto', origin='lower')
plt.xlabel('Time bin')
plt.ylabel('Frequency bin')
plt.colorbar()
plt.savefig('stretchedPaddedData.png')