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

import os

log = True
log_fitted_objects = True
log_statistics = True

maximum_planar_offset = 1e-6
point_transform_check_tolerance = 1e-3  # always in mm!!!

# Output units for circle parameters
output_units = {
    "distances": "um",  # Choose from ["m", "cm", "mm", "um"]
    "angles": "gon"   # Choose from ["gon", "rad", "mrad", "deg"]
}

output_units_LT = {
    "distances": "mm",  # Choose from ["m", "cm", "mm", "um"]
    "angles": "gon"   # Choose from ["gon", "rad", "mrad", "deg"]
}

# Define the log precision dictionary in your configuration file
log_precision = {
    'distances': (0.1, 'um'),
    'angles': (0.0001, 'gon')
}

log_filename = "Test_logs.txt"

# Get the current directory (the directory where the script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the relative path to the data folder
data_folder = "Data"
log_folder = "Logs"

# Set the name of the data file
data_file = "Points_circle_test.txt"
keyence1 = "Keyence_meas_python_input.txt"
keyence2 = "Keyence_meas_python_input2d.txt"
LT = "LT_meas_python_input.txt"
planeANDcircle_test_points = "points_to_fit.txt"
test_plane1 = "Plane_test_1.txt"

# Create the full file path by joining the current directory
# and the subfolder name
logfile_path = os.path.join(current_dir, log_folder, log_filename)
circle_file = os.path.join(current_dir, data_folder, data_file)
keyence_path = os.path.join(current_dir, data_folder, keyence1)
keyence2d_path = os.path.join(current_dir, data_folder, keyence2)
LT_path = os.path.join(current_dir, data_folder, LT)
test_plane1_path = os.path.join(current_dir, data_folder, "Plane_test_1.txt")

planeANDcircle_test_points_path = os.path.join(current_dir, data_folder,
                                               planeANDcircle_test_points)

if log and log_statistics:
    log_statistics = True
else: log_statistics = False

if log:
    log = (logfile_path, log_precision)

"""There will have to be a list of referenced angles between
    the systems of Keyence system and the outside world. It
    will be stored in a dictionary here in config file."""
