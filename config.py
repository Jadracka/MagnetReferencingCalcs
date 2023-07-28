# config.py

# Output units for circle parameters
output_units = {
    "center": "mm",  # Choose from ["m", "cm", "mm"]
    "radius": "mm"   # Choose from ["m", "cm", "mm"]
}

import os

# Get the current directory (the directory where the script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the relative path to the data folder
data_folder = "Data"

# Set the name of the data file
data_file = "Points_circle_test.txt"

# Create the full file path by joining the current directory and the subfolder name
circle_file = os.path.join(current_dir, data_folder, data_file)