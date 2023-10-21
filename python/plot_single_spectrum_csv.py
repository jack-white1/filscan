# a python script that loads single_spectrum_dedispersed.csv
# the csv is in the format of 2 columns: real and imaginary components
# it plots the magnitude of the complex spectrum and saves it to a png file
# the png file is saved in the same directory as the csv file
# the png file name is the same as the csv file name
# the script takes one argument: the csv file name

import sys
import numpy as np
import matplotlib.pyplot as plt

# check if the number of arguments is correct
if len(sys.argv) != 2:
    print("Usage: python plot_single_spectrum_csv.py <csv_file_name>")
    sys.exit(1)
    
# read the csv file
csv_file_name = sys.argv[1]
csv_file = open(csv_file_name, "r")
csv_file_lines = csv_file.readlines()
csv_file.close()

# extract the real and imaginary components
real = []
imag = []
for line in csv_file_lines:
    line = line.strip()
    if line == "":
        continue
    line = line.split(",")
    real.append(float(line[0]))
    imag.append(float(line[1]))

# convert the real and imaginary components to complex numbers
complex_spectrum = np.array(real) + 1j * np.array(imag)

# plot the magnitude of the complex spectrum on a log scale
plt.plot(np.abs(complex_spectrum))
plt.yscale("log")
plt.xlabel("Frequency channel")
plt.ylabel("Magnitude")
plt.title(csv_file_name)
plt.savefig(csv_file_name + ".png")
plt.close()
