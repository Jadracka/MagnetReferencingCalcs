# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 2023

@author: jbarker
"""

"""

   _____             __ _                       _   _                __ _ _      
  / ____|           / _(_)                     | | (_)              / _(_) |     
 | |     ___  _ __ | |_ _  __ _ _   _ _ __ __ _| |_ _  ___  _ __   | |_ _| | ___ 
 | |    / _ \| '_ \|  _| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \  |  _| | |/ _ \
 | |___| (_) | | | | | | | (_| | |_| | | | (_| | |_| | (_) | | | | | | | | |  __/
  \_____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_| |_| |_|_|\___|
                           __/ |                                                 
                          |___/                                                  
"""

# Output units for circle parameters
output_units = {
    "distances": "mm",  # Choose from ["m", "cm", "mm", "um"]
    "angles": "gon"   # Choose from ["gon", "rad", "mrad", "deg"]
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