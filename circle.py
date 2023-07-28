# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:38:29 2023

@author: jbarker, ChatGPT3.5
"""

#import numpy as np
#from scipy.optimize import least_squares
import config as cg # Import the config.py file
import functions as fc



data_file1 = "points_format_cartesian.txt"
circle_parameters = fc.fit_circle_3d(cg.circle_file)

if circle_parameters is not None:
    center_x, center_y, center_z, radius = circle_parameters
    print("Circle Center (x, y, z):", (center_x, center_y, center_z))
    print("Circle Radius (mm):", radius)
else:
    print("No circle could be fit to the data.")


"""
# Example usage with format: Cartesian system (points_format_cartesian.txt)
data_file1 = "points_format_cartesian.txt"
circle_parameters1 = fit_circle_3d(data_file1)

if circle_parameters1 is not None:
    center_x, center_y, center_z, radius = circle_parameters1
    print("Circle Center (x, y, z):", (center_x, center_y, center_z))
    print("Circle Radius (mm):", radius)
else:
    print("No circle could be fit to the data.")



# Example usage with format: Spherical system (points_format_spherical.txt)
data_file2 = "points_format_spherical.txt"
circle_parameters2 = fit_circle_3d(data_file2)

if circle_parameters2 is not None:
    center_x, center_y, center_z, radius = circle_parameters2
    print("Circle Center (x, y, z):", (center_x, center_y, center_z))
    print("Circle Radius (mm):", radius)
else:
    print("No circle could be fit to the data.")
"""