# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:54:12 2023

@author: jbarker
"""

import numpy as np
from scipy.optimize import least_squares

def gon_to_radians(gon):
    return gon * (2 * np.pi / 400)

def degrees_to_radians(degrees):
    return degrees * (np.pi / 180.0)

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

def spherical_to_cartesian_unit(distance, angle_unit, azimuth, zenith_angle, d_unit):
    # Convert the angle to radians if needed
    if angle_unit == 'gons':
        azimuth = gon_to_radians(azimuth)
        zenith_angle = gon_to_radians(zenith_angle)
    elif angle_unit == 'degrees':
        azimuth = degrees_to_radians(azimuth)
        zenith_angle = degrees_to_radians(zenith_angle)

    # Convert distance to millimeters
    if d_unit == 'um':
        distance *= 0.001
    elif d_unit == 'mm':
        distance *= 1.0
    elif d_unit == 'cm':
        distance *= 10.0
    elif d_unit == 'm':
        distance *= 1000.0
    else:
        raise ValueError("Invalid distance unit specified.")

    # Calculate Cartesian coordinates in millimeters
    x = distance * np.sin(zenith_angle) * np.cos(azimuth)
    y = distance * np.sin(zenith_angle) * np.sin(azimuth)
    z = distance * np.cos(zenith_angle)

    return x, y, z, 'mm'

def coordinate_unit_to_mm(unit):
    # Convert the coordinate unit to mm
    if unit == "um":
        return 0.001
    elif unit == "mm":
        return 1.0
    elif unit == "cm":
        return 10.0
    elif unit == "m":
        return 1000.0
    else:
        raise ValueError("Invalid coordinate unit specified.")

def circle_residuals(params, x, y, z):
    cx, cy, cz, r = params
    return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 - r**2

def generate_noisy_ellipse_points(a, b, center_x, center_y, num_points, std_dev):
    """
    Example usage:
    a = 3.0  # Semi-major axis
    b = 1.5  # Semi-minor axis
    center_x = 2.0  # x-coordinate of the center
    center_y = 1.0  # y-coordinate of the center
    num_points = 100  # Number of points to generate
    std_dev = 0.1  # Standard deviation for the noise (adjust as desired)
    """
    # Generate angles evenly spaced around the ellipse (0 to 2*pi)
    angles = np.linspace(0, 2*np.pi, num_points)

    # Parametric equations of the ellipse
    x = center_x + a * np.cos(angles)
    y = center_y + b * np.sin(angles)

    # Add noise from a standard normal distribution
    noise_x = np.random.normal(0, std_dev, num_points)
    noise_y = np.random.normal(0, std_dev, num_points)

    x += noise_x
    y += noise_y
    
    """
    print("True Parameters:")
    print("Semi-Major Axis (a):", a)
    print("Semi-Minor Axis (b):", b)
    print("Center (x, y):", center_x, center_y)
    """
    
    return x, y

def generate_ellipse_points(center, semi_major_axis, semi_minor_axis, num_points):
    """
    # Example usage:
    Center = (3, 4)  # Center coordinates of the ellipse
    Semi_major_axis = 5  # Length of the semi-major axis
    Semi_minor_axis = 3  # Length of the semi-minor axis
    Num_points = 100  # Number of points to generate on the ellipse
    
    x_points, y_points = generate_ellipse_points(Center, Semi_major_axis, Semi_minor_axis, Num_points)
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    x_points = center[0] + semi_major_axis * cos_angles
    y_points = center[1] + semi_minor_axis * sin_angles

    return x_points, y_points

def get_unit_multiplier(unit):
    if unit == "m":
        return 1000.0
    elif unit == "cm":
        return 10.0
    elif unit == "mm":
        return 1.0
    elif unit == "um":
        return 0.001
    else:
        raise ValueError("Invalid unit specified in the header.")
        
def read_data_from_file(file_path):
    data_dict = {}
    point_ids_set = set()

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Get the data format from the header
        data_format_line = lines[1].split(':')
        data_format = data_format_line[1].strip().lower()

        # Get the units from the header
        units_line = lines[2].split(':')
        units = units_line[1].strip().split()

        if data_format == 'spherical':
            angle_unit = units[0]
            d_unit = units[1]
            coordinate_unit = None
        elif data_format == 'cartesian':
            angle_unit = None
            d_unit = None
            coordinate_unit = units[0]
        else:
            raise ValueError("Invalid data format specified in the header.")

        # Process the data lines
        for line_num, line in enumerate(lines[4:], start=5):
            line = line.strip().split()
            PointID = line[0]

            if PointID in point_ids_set:
                raise ValueError(f"Duplicate PointID found in the data. Line number: {line_num}")
            else:
                point_ids_set.add(PointID)

            if data_format == 'spherical':
                azimuth = float(line[1].replace(',', '.'))
                zenith_angle = float(line[2].replace(',', '.'))
                distance = float(line[3].replace(',', '.'))

                # Convert spherical to Cartesian
                x, y, z, coordinate_unit = spherical_to_cartesian(distance, angle_unit, azimuth, zenith_angle, d_unit)

            elif data_format == 'cartesian':
                x = float(line[1].replace(',', '.'))
                y = float(line[2].replace(',', '.'))
                z = float(line[3].replace(',', '.'))

            else:
                raise ValueError("Invalid data format specified in the header.")

            # Store data in the dictionary
            data_dict[PointID] = {
                'Hz': azimuth if azimuth is not None else None,
                'V': zenith_angle if zenith_angle is not None else None,
                'd': distance if distance is not None else None,
                'X': x,
                'Y': y,
                'Z': z,
                'angle_unit': angle_unit,
                'd_unit': d_unit,
                'coordinate_unit': coordinate_unit
            }

    return data_dict



def fit_circle_3d(data_dict, output_units):
    # Extract data from the dictionary
    x, y, z = [], [], []

    for point_data in data_dict.values():
        # Parse data in Cartesian format
        x.append(point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
        y.append(point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
        z.append(point_data['Z'] * coordinate_unit_to_mm(point_data['coordinate_unit']))

    # Fit a circle in 3D using least squares optimization
    initial_guess = np.array([np.mean(x), np.mean(y), np.mean(z), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2 + (z - np.mean(z))**2))])
    result = least_squares(circle_residuals, initial_guess, args=(x, y, z))

    # Check if the optimization succeeded
    if result.success:
        center_x, center_y, center_z, radius = result.x
    else:
        print("No circle could be fit to the data.")
        return None

    # Scale the output based on output_units from config.py
    center_scale = 1.0
    radius_scale = 1.0

    if output_units["center"] == "m":
        center_scale = 0.001
        radius_scale = 0.001
    elif output_units["center"] == "cm":
        center_scale = 0.01
        radius_scale = 0.01
    elif output_units["center"] == "mm":
        center_scale = 1.0
        radius_scale = 1.0
    elif output_units["center"] == "um":
        center_scale = 1000.0
        radius_scale = 1000.0
   

    # Scale the output
    center_x, center_y, center_z, radius = center_x * center_scale, center_y * center_scale, center_z * center_scale, radius * radius_scale

    return center_x, center_y, center_z, radius