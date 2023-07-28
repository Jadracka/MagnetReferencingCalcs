# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:54:12 2023

@author: jbarker
"""

import numpy as np
from scipy.optimize import least_squares
import config as cg

def gon_to_radians(gon):
    return gon * (2 * np.pi / 400)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def circle_residuals(params, x, y, z):
    cx, cy, cz, r = params
    return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 - r**2

def fit_circle_3d(data_file):
    with open(data_file, 'r') as file:
        lines = file.readlines()

        # Get the data format from the header
        header_lines = lines[:5]
        data_format_line = header_lines[1].split(':')
        data_format = data_format_line[1].strip().lower()
        
        # Detect decimal separator from the data lines
        comma_found = False
        for line in lines[5:]:
            line = line.strip().split()
            for val in line[1:]:
                if ',' in val:
                    comma_found = True
                    break
            if comma_found:
                break

        # Set the separator for parsing data based on the detection
        data_separator = ',' if comma_found else '.'

        x, y, z = [], [], []
        for line in lines[5:]:
            line = line.strip().split()
#            point_number = line[0]
            if data_format == 'cartesian':
                # Parse data in Cartesian format
                x.append(float(line[1].replace(data_separator, '.')))
                y.append(float(line[2].replace(data_separator, '.')))
                z.append(float(line[3].replace(data_separator, '.')))
            elif data_format == 'spherical':
                # Parse data in spherical format
                azimuth = float(line[1].replace(data_separator, '.'))
                zenith_angle = float(line[2].replace(data_separator, '.'))
                distance = float(line[3].replace(data_separator, '.'))

                # Convert spherical to Cartesian
                x_i, y_i, z_i = spherical_to_cartesian(distance, gon_to_radians(azimuth), gon_to_radians(zenith_angle))

                x.append(x_i)
                y.append(y_i)
                z.append(z_i)
            else:
                raise ValueError("Invalid data format specified in the header.")

        # Convert lists to NumPy arrays
        x, y, z = np.array(x), np.array(y), np.array(z)

        # Fit a circle in 3D using least squares optimization
        initial_guess = np.array([np.mean(x), np.mean(y), np.mean(z), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2 + (z - np.mean(z))**2))])
        result = least_squares(circle_residuals, initial_guess, args=(x, y, z))

                # Scale the output based on output_units from config.py

        if result.success:
            center_x, center_y, center_z, radius = result.x
        else:
            print("No circle could be fit to the data.")
            return None

        # Scale the output based on output_units from config.py
        center_scale = 1.0
        radius_scale = 1.0

        if cg.output_units["center"] == "m":
            center_scale = 0.001
        elif cg.output_units["center"] == "cm":
            center_scale = 0.01
        elif cg.output_units["center"] == "mm":
            center_scale = 1.0
        else:
            raise ValueError("Invalid center units specified in config.py.")

        if cg.output_units["radius"] == "m":
            radius_scale = 0.001
        elif cg.output_units["radius"] == "cm":
            radius_scale = 0.01
        elif cg.output_units["radius"] == "mm":
            radius_scale = 1.0
        else:
            raise ValueError("Invalid radius units specified in config.py.")

        # Scale the output
        center_x, center_y, center_z, radius = center_x * center_scale, center_y * center_scale, center_z * center_scale, radius * radius_scale

        return center_x, center_y, center_z, radius