# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:54:12 2023

@author: jbarker
"""

import numpy as np
from scipy.optimize import least_squares
import datetime
import inspect
import os


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

def circle_residuals_3d(params, x, y, z):
    cx, cy, cz, r = params
    return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 - r**2

def get_variable_name(variable):
    """# Example usage:
        name = "John"
        age = 30

    variable_name_as_string = get_variable_name(name)
    print(variable_name_as_string) 
    prints 'name' """
    # Get the calling frame
    frame = inspect.currentframe().f_back
    
    # Find the variable name by checking the locals and globals dictionaries
    for name, value in frame.f_locals.items():
        if value is variable:
            return name
    for name, value in frame.f_globals.items():
        if value is variable:
            return name

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
        elif data_format == 'cartesian2d':  # New format for 2D Cartesian data
            angle_unit = None
            d_unit = None
            coordinate_unit = units[0]
        else:
            raise ValueError("Invalid data format specified in the header.")

        # Process the data lines
        line_number = 0
        for line in lines:
            line_number += 1
            if not line.startswith('#'):
                line = line.strip().split()

                # Skip empty lines
                if not line:
                    continue

                PointID = line[0]

                if data_format == 'spherical':
                    azimuth = float(line[1].replace(',', '.'))
                    zenith_angle = float(line[2].replace(',', '.'))
                    distance = float(line[3].replace(',', '.'))

                    # Convert spherical to Cartesian
                    x, y, z, coordinate_unit = spherical_to_cartesian_unit(distance, angle_unit, azimuth, zenith_angle, d_unit)

                elif data_format == 'cartesian' or data_format == 'cartesian2d':
                    x = float(line[1].replace(',', '.'))
                    y = float(line[2].replace(',', '.'))
                    z = None if len(line) < 4 else float(line[3].replace(',', '.'))  # Z coordinate for 3D Cartesian, None for 2D Cartesian
                # Check for duplicate PointIDs
                if PointID in data_dict:
                    raise ValueError(f"Duplicate PointID '{PointID}' found in line {line_number}.")

                # Store data in the dictionary
                data_dict[PointID] = {
                    'Hz': azimuth if data_format == 'spherical' else None,
                    'V': zenith_angle if data_format == 'spherical' else None,
                    'd': distance if data_format == 'spherical' else None,
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'angle_unit': angle_unit if data_format == 'spherical' else None,
                    'd_unit': d_unit if data_format == 'spherical' else None,
                    'coordinate_unit': coordinate_unit,

                }

    return data_dict, file_path


def get_angle_scale(output_units):
    if output_units["angles"] == "gon":
        return np.pi / 200.0  # Convert gon to radians
    elif output_units["angles"] == "rad":
        return 1.0
    elif output_units["angles"] == "mrad":
        return 0.001
    elif output_units["angles"] == "deg":
        return np.pi / 180.0  # Convert degrees to radians
    else:
        raise ValueError(f"Invalid angle unit '{output_units['angles']}' specified.")



def fit_circle_3d(data_tuple, output_units):
    data_dict, file_path = data_tuple
    # Extract data from the dictionary
    x, y, z = [], [], []

    for point_data in data_dict.values():
        # Parse data in Cartesian format
        x.append(point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
        y.append(point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
        z.append(point_data['Z'] * coordinate_unit_to_mm(point_data['coordinate_unit']))

    # Fit a circle in 3D using least squares optimization
    initial_guess = np.array([np.mean(x), np.mean(y), np.mean(z), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2 + (z - np.mean(z))**2))])
    result = least_squares(circle_residuals_3d, initial_guess, args=(x, y, z))

    # Check if the optimization succeeded
    if result.success:
        center_x, center_y, center_z, radius = result.x
    else:
        print("No circle could be fit to the data. Optimization process failed.")
        return None

    # Scale the output based on output_units from config.py
    result_scale = get_distance_scale(output_units)
    
    # Prepare statistics if requested
    statistics = None
    residuals = circle_residuals_3d(result.x, x, y, z)
    statistics = {
            "Standard Deviation": np.std(residuals),
            "Maximum Residual": np.max(abs(residuals)),
            "Minimum Residual": np.min(abs(residuals)),
            "RMS": np.sqrt(np.mean(residuals**2)),
            "Mean": np.mean(residuals),
            "Median": np.median(residuals)
        }

    # Scale the output
    center_x, center_y, center_z, radius = center_x * result_scale, center_y * result_scale, center_z * result_scale, radius * result_scale

    return {"center_x": center_x, "center_y": center_y, "center_z": center_z, "radius": radius, "statistics": statistics}

def fit_circle_2d(data_tuple, output_units, log_file_path=None):
    data_dict, file_path = data_tuple
    # Extract data from the dictionary
    x, y = [], []

    for point_data in data_dict.values():
        # Parse data in Cartesian format
        x.append(point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
        y.append(point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit']))

    # Fit a circle in 2D using least squares optimization
    initial_guess = np.array([np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2))])
    result = least_squares(circle_residuals_2d, initial_guess, args=(x, y))

    # Check if the optimization succeeded
    if result.success:
        center_x, center_y, radius = result.x
    else:
        print("No circle could be fit to the data.")
        return None

    # Scale the output based on output_units from config.py
    result_scale = get_distance_scale(output_units)

    # Scale the output
    center_x, center_y, radius = center_x * result_scale, center_y * result_scale, radius * result_scale
    
    # Prepare statistics if requested
    statistics = None
    
    residuals = circle_residuals_2d(result.x, x, y)
    statistics = {
            "Standard Deviation": np.std(residuals),
            "Maximum |Residual|": np.max(abs(residuals)),
            "Minimum |Residual|": np.min(abs(residuals)),
            "RMS": np.sqrt(np.mean(residuals**2)),
            "Mean": np.mean(residuals),
            "Median": np.median(residuals)
        }
    if log_file_path:
        write_circle_fit_log(log_file_path, file_path, data_dict, {"center_x": center_x, "center_y": center_y}, radius, output_units, statistics, log_statistics=False)

    return {"center": (center_x, center_y), "radius": radius, "statistics": statistics}


# Define the circle residuals function for 2D
def circle_residuals_2d(params, x, y):
    center_x, center_y, radius = params
    return np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius

def get_distance_scale(output_units):
    if output_units["distances"] == "m":
        return 0.001
    elif output_units["distances"] == "cm":
        return 0.01
    elif output_units["distances"] == "mm":
        return 1.0
    elif output_units["distances"] == "um":
        return 1000.0
    else:
        raise ValueError(f"Invalid distance unit '{output_units['distances']}' specified.")

def plane_residuals(coeffs, x, y, z):
    a, b, c = coeffs
    return a * np.array(x) + b * np.array(y) + c - np.array(z)

def fit_plane_3d(data_tuple, output_units, log_statistics):
    data_dict, file_path = data_tuple
    # Check if 3D points are available
    if not any('Z' in point_data for point_data in data_dict.values()):
        raise ValueError("The input data are not in 3D, a plane cannot be fit through those points.")

    # Extract 3D points from the dictionary
    x, y, z = [], [], []

    for point_data in data_dict.values():
        if 'X' in point_data and 'Y' in point_data and 'Z' in point_data:
            x.append(point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
            y.append(point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
            z.append(point_data['Z'] * coordinate_unit_to_mm(point_data['coordinate_unit']))

    if len(x) < 3:
        raise ValueError("Insufficient 3D points available to fit a plane. At least three 3D points (X, Y, Z) are required for plane fitting.")

    # Fit a plane using least squares optimization
    initial_guess = np.array([1.0, 1.0, 1.0])
    result = least_squares(plane_residuals, initial_guess, args=(x, y, z))

    # Check if the optimization succeeded
    if not result.success:
        raise ValueError("Failed to fit a plane to the data. Please check the input data_dict and try again.")

    # Scale the output based on output_units
    result_scale = get_distance_scale(output_units)

    # Scale the coefficients
    a, b, c = result.x * result_scale
    
    # Calculate the plane offset D
    d = -1.0 * np.mean(a * np.array(x) + b * np.array(y) + c * np.array(z))

    return (a, b, c,d)
"""
# Example usage with data_dict and output_units as inputs
try:
    plane_coefficients = fit_plane_3d(data_dict, output_units)
    print("Plane coefficients (a, b, c):", plane_coefficients)
except ValueError as e:
    print(str(e))
"""


def write_circle_fit_log(log_file_path, file_path, data_dict, center, radius, output_units, statistics, log_statistics=False):
    if not log_file_path:
        return
    
    with open(log_file_path, 'a+') as log_file:
        circle_name = os.path.basename(file_path)
        # Write header with date and time of calculation
        log_file.write("Circle {} Fitting Results:\n".format(circle_name))
        log_file.write("Calculation Date: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        log_file.write("Source File: {}\n".format(file_path))
        log_file.write("Units: {}\n".format(output_units["distances"]))
        log_file.write("\n")

        # Write circle fitting results
        if "center_z" in center:
            log_file.write("Center: {},{},{}\n".format(center["center_x"],center["center_y"],center["center_z"]))
        else:
            log_file.write("Center: {},{}\n".format(center["center_x"],center["center_y"]))
 
        log_file.write("Radius: {}\n".format(radius))
        log_file.write("\n")

        # Write statistics if available and log_statistics is True
        if log_statistics:
            log_file.write("Best-fit Statistics:\n")
            for key, value in statistics.items():
                log_file.write("{}: {}\n".format(key, value))
            log_file.write("\n")