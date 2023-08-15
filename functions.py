"""
Functions.

Functions for Magnet Referencing Code.

Author: Jana Barker
Date: Created on Fri Jul 28 13:54:12 2023

Description:
------------
This is a file containing all functions necessary in orther to fit circles into
measured 3D or 2D points.
This file also contains functions to read in a Spatial Analyzer output file
and pars it into needed dictionaries.

Note:
----
- made together with ChatGPT, but tested by human.
"""


import numpy as np
from scipy.optimize import least_squares
import datetime
import inspect
import os
import matplotlib.pyplot as plt
import random
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from typing import Union


def gon_to_radians(gon):
    """
    Convert an angle in gradians (gons) to radians.

    This function takes an angle in gradians and returns its equivalent
    angle in radians.

    Parameters
    ----------
    gon (float): Angle in gradians (gons) to be converted to radians.

    Returns
    -------
    float: Angle in radians equivalent to the input angle in gradians.

    Examples
    --------
    >>> gon_to_radians(200)
    3.141592653589793
    >>> gon_to_radians(100)
    1.5707963267948966
    """
    return gon * (2 * np.pi / 400)


def degrees_to_radians(degrees):
    """
    Convert an angle in degrees to radians.

    This function takes an angle in degrees and returns its equivalent
    angle in radians.

    Parameters
    ----------
    degrees (float): Angle in degrees to be converted to radians.

    Returns
    -------
    float: Angle in radians equivalent to the input angle in degrees.

    Examples
    --------
    >>> degrees_to_radians(180)
    3.141592653589793
    >>> degrees_to_radians(90)
    1.5707963267948966
    """
    return degrees * (np.pi / 180.0)


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Converts the given Cartesian coordinates (x, y, z) to spherical coordinates
    (r, theta, phi).
    r represents the radial distance from the origin to the point.
    theta represents the inclination angle measured from the positive z-axis.
    phi represents the azimuthal angle measured from the positive x-axis.

    Parameters
    ----------
    x (float): x-coordinate in Cartesian space.
    y (float): y-coordinate in Cartesian space.
    z (float): z-coordinate in Cartesian space.

    Returns
    -------
    tuple: A tuple containing (r, theta, phi) in spherical coordinates.

    Examples
    --------
    >>> cartesian_to_spherical(1, 1, 1)
    (1.7320508075688772, 0.9553166181245093, 0.7853981633974483)
    >>> cartesian_to_spherical(0, 0, 2)
    (2.0, 1.5707963267948966, 0.0)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(d, Hz, V):
    """
    Convert spherical coordinates to 3D Cartesian coordinates.

    Converts the given spherical coordinates (d, Hz, V) to 3D Cartesian
    coordinates (X, Y, Z).
    d represents the radial distance from the origin to the point.
    Hz represents the horizontal angle measured from the positive x-axis.
    V represents the zenithal angle measured from the positive z-axis.

    Parameters
    ----------
    d (float): Radial distance in spherical coordinates.
    Hz (float): Horizontal angle in radians.
    V (float): Zenithal angle in radians.

    Returns
    -------
    tuple: A tuple containing (X, Y, Z) in 3D Cartesian coordinates.

    Examples
    --------
    >>> spherical_to_cartesian(1.0, 0.7853981633974483, 0.7853981633974483)
    (0.5000000000000001, 0.5000000000000001, 0.49999999999999994)
    >>> spherical_to_cartesian(2.0, 1.5707963267948966, 0.0)
    (1.2246467991473532e-16, 2.0, 0.0)
    """
    X = d * np.cos(-Hz) * np.sin(V)
    Y = d * np.sin(-Hz) * np.sin(V)
    Z = d * np.cos(V)
    return X, Y, Z


def spherical_to_cartesian_unit(d, d_unit, angle_unit, Hz, V):
    """
    Convert Spherical to Cartesian coordinates with specified units.

    Converts the given spherical coordinates (d, Hz, V) to 3D Cartesian
    coordinates (x, y, z) in the specified units.

    Parameters
    ----------
    d (float): Radial distance in spherical coordinates.
    angle_unit (str): Unit of the angles. Choose from 'rad', 'gon', or 'deg'.
    Hz (float): Horizontal angle.
    V (float): Zenithal angle.
    d_unit (str): Unit of the radial distance.
        Choose from 'um', 'mm', 'cm', or 'm'.

    Returns
    -------
    tuple: A tuple containing (x, y, z) Cartesian coordinates in the specified
    units.

    Raises
    ------
    ValueError: If an invalid angle unit or radial distance unit is specified.

    Examples
    --------
    >>> spherical_to_cartesian_unit(1.0, 'rad', 0.7853981633974483,
                                    0.7853981633974483, 'mm')
    (0.5000000000000001, 0.5000000000000001, 0.49999999999999994)
    >>> spherical_to_cartesian_unit(2.0, 'deg', 90, 0.0, 'cm')
    (1.2246467991473532e-16, 200.0, 0.0, 'mm')
    """
    # Convert the angle to radians if needed
    if angle_unit == 'gon':
        Hz = gon_to_radians(Hz)
        V = gon_to_radians(V)
    elif angle_unit == 'deg':
        Hz = degrees_to_radians(Hz)
        V = degrees_to_radians(V)

    # Convert d to millimeters
    if d_unit == 'um':
        d *= 0.001
    elif d_unit == 'mm':
        d *= 1.0
    elif d_unit == 'cm':
        d *= 10.0
    elif d_unit == 'm':
        d *= 1000.0
    else:
        raise ValueError("Invalid d unit specified.")

    # Calculate Cartesian coordinates in millimeters
    x = d * np.sin(V) * np.cos(-Hz)
    y = d * np.sin(V) * np.sin(-Hz)
    z = d * np.cos(V)

    return x, y, z, 'mm'


def coordinate_unit_to_mm(unit):
    """
    Convert a coordinate unit to millimeters.

    Converts a coordinate unit to its equivalent value in millimeters.

    Parameters
    ----------
    unit : str
        The coordinate unit to be converted. Should be one of "um", "mm", "cm",
        or "m".

    Returns
    -------
    float
        The equivalent value of the coordinate unit in millimeters.

    Raises
    ------
    ValueError
        If the input coordinate unit is not one of "um", "mm", "cm", or "m".

    Examples
    --------
    >>> coordinate_unit_to_mm("mm")
    1.0
    >>> coordinate_unit_to_mm("cm")
    10.0
    >>> coordinate_unit_to_mm("m")
    1000.0
    """
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


def get_variable_name(variable):
    """
    Get the name of a variable.

    This function takes a variable and returns its name as a string. It does
    so by examining the calling frame's locals and globals dictionaries.

    Parameters
    ----------
    variable : any
        The variable for which the name is to be retrieved.

    Returns
    -------
    str
        The name of the given variable as a string.

    Examples
    --------
    Example usage:
    >>> name = "John"
    >>> age = 30
    >>> variable_name_as_string = get_variable_name(name)
    >>> print(variable_name_as_string)
    'name'
    """
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
    """Generate points along a noisy ellipse.

    Parameters
    ----------
    a : float
        Semi-major axis of the ellipse.
    b : float
        Semi-minor axis of the ellipse.
    center_x : float
        x-coordinate of the center of the ellipse.
    center_y : float
        y-coordinate of the center of the ellipse.
    num_points : int
        Number of points to generate.
    std_dev : float
        Standard deviation for the noise added to the points.

    Returns
    -------
    x : numpy.ndarray
        Array of x-coordinates of the generated points.
    y : numpy.ndarray
        Array of y-coordinates of the generated points.

    Example
    -------
    a = 3.0
    b = 1.5
    center_x = 2.0
    center_y = 1.0
    num_points = 100
    std_dev = 0.1
    x, y = generate_noisy_ellipse_points(a, b, center_x, center_y,
                                         num_points, std_dev)
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
                    x, y, z, coordinate_unit = spherical_to_cartesian_unit(distance, d_unit, angle_unit, azimuth, zenith_angle)

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


def make_residual_stats(residuals: Union[np.ndarray,list,tuple]):
    if not isinstance(residuals, np.ndarray):
        residuals = np.array(residuals)
        
    statistics = {
        "Standard Deviation": np.std(residuals),
        "Maximum |Residual|": np.max(abs(residuals)),
        "Minimum |Residual|": np.min(abs(residuals)),
        "RMS": np.sqrt(np.mean(residuals**2)),
        "Mean": np.mean(residuals),
        "Median": np.median(residuals)
    }
    return statistics


def fit_circle_2d(data_tuple, output_units, log_file_path=None):
    data_dict, file_path = data_tuple
    circle_name = os.path.splitext(os.path.basename(file_path))[0]
    # Extract data from the dictionary
    x, y = [], []

    # Create a list to store point names
    point_names = []

    for point_name, point_data in data_dict.items():
        # Parse data in Cartesian format
        x.append(point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
        y.append(point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit']))

        # Store the point name
        point_names.append(point_name)

    # Fit a circle in 2D using least squares optimization
    initial_guess = np.array([np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2))])
    result = least_squares(circle_residuals_2d, initial_guess, args=(x, y))

    # Check if the optimization succeeded
    if result.success:
        center_x, center_y, radius = result.x
    else:
        print("No circle  could be fit to the data of {circle_name}.")
        return None

    # Scale the output based on output_units from config.py
    result_scale = get_distance_scale(output_units)

    # Scale the output
    center_x, center_y, radius = center_x * result_scale, center_y * result_scale, radius * result_scale

    # Calculate radial offsets
    radial_offsets = np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius

    # Create a dictionary to store radial offsets for each point
    point_radial_offsets = {point_name: offset for point_name, offset in zip(point_names, radial_offsets)}

    # Prepare statistics if requested
    statistics = make_residual_stats(radial_offsets)

    if log_file_path:
        write_circle_fit_log(log_file_path, file_path, data_dict, {"center_x": center_x, "center_y": center_y}, radius, output_units, statistics, log_statistics=False)

    return {"name": circle_name, "center": (center_x, center_y), "radius": radius, "statistics": statistics, "point_radial_offsets": point_radial_offsets}

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
        raise ValueError(f"Invalid distance unit '{output_units['distances']}'" 
                         f" specified.")


def fit_plane(data_tuple, output_units, log_statistics='False'):
    data_dict, file_path = data_tuple
    plane_name = os.path.splitext(os.path.basename(file_path))[0]

    # Check if 3D points are available
    if not any('Z' in point_data for point_data in data_dict.values()):
        raise ValueError(f"The input data are not in 3D, a {plane_name} plane "
                         f"cannot be fit through those points.")

    # Extract 3D points from the dictionary
    x, y, z = [], [], []

    for point_data in data_dict.values():
        if 'X' in point_data and 'Y' in point_data and 'Z' in point_data:
            x.append(point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
            y.append(point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit']))
            z.append(point_data['Z'] * coordinate_unit_to_mm(point_data['coordinate_unit']))

    if len(x) < 3:
        raise ValueError(f"Insufficient 3D points available to fit a plane in "
                         f"{plane_name}. At least three 3D points (X, Y, Z) "
                         f"are required for plane fitting.")

    # Calculate the centroid of the points
    centroid_x, centroid_y, centroid_z = np.mean(x), np.mean(y), np.mean(z)

    # Shift the points to the centroid
    x_shifted = x - centroid_x
    y_shifted = y - centroid_y
    z_shifted = z - centroid_z

    # Stack the coordinates as a matrix
    A = np.column_stack((x_shifted, y_shifted, z_shifted))

    # Use Singular Value Decomposition (SVD) to fit the plane
    _, _, V = np.linalg.svd(A, full_matrices=False)
    normal = V[-1]  # The normal vector of the plane is the last column of V

    # Extract coefficients from the normal vector
    a, b, c = normal

    # Calculate d coefficient
    d = -(a * centroid_x + b * centroid_y + c * centroid_z)

    # Calculate the angle of the plane with respect to coordinate axes
#    angles = get_plane_angles(normal, output_units)

    # Scale the output based on output_units
#    angle_scale = get_angle_scale(output_units)

    # Scale the coefficients and angles
#    a, b, c = a * result_scale, b * result_scale, c * result_scale
#    angles = [angle * angle_scale for angle in angles]
    
    
    # Calculate residuals 
    plane_params = (a, b, c, d)
    residual_dict = {point_name: point_to_plane_distance(X, Y, Z, plane_params)
                     for point_name, X, Y, Z in zip(list(data_dict.keys()), x, y, z) 
                     }

    # Prepare statistics if requested
    statistics = make_residual_stats(np.array(list(residual_dict.values())))
    
    return plane_params, {'planer_offsets': residual_dict, 
                          'plane_statistics': statistics}


def get_plane_angles(normal_vector, output_units):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    Rx = np.arctan2(normal_vector[0], normal_vector[1])
    Ry = np.arctan2(normal_vector[1], normal_vector[2])
    Rz = np.arctan2(normal_vector[2], normal_vector[0])
        # Convert angles to the desired output units
    angle_scale = get_angle_scale(output_units)
    Rx *= angle_scale
    Ry *= angle_scale
    Rz *= angle_scale
    return Rx, Ry, Rz


def write_circle_fit_log(log_file_path, file_path, data_dict, center, radius, output_units, statistics, log_statistics=False):
    if not log_file_path:
        return

    with open(log_file_path, 'a+') as log_file:
        circle_name = os.path.basename(file_path)
        # Write header with date and time of calculation
        log_file.write("_" * 100)
        log_file.write("\nCircle {} Fitting Results:\n".format(circle_name))
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
    return True


def check_plane_projection(points, center, normal_vector, tolerance):
    """
    Check if the points lie on the fitted plane or need to be projected onto it.

    Parameters:
        points (list of tuples): List of (x, y, z) points of the circle.
        center (tuple): Center of the fitted circle in (x, y, z) coordinates.
        normal_vector (tuple): Normal vector of the fitted plane in (a, b, c) form.
        tolerance (float): Maximum allowable residual to consider points lying on the plane.

    Returns:
        bool: True if points lie on the plane, False if points need to be projected onto the plane.
    """
    # Calculate the residuals of the points from the fitted circle
    residuals = [np.dot(np.array(point) - np.array(center), np.array(normal_vector)) for point in points]

    # Check if the maximum residual is within the tolerance
    if max(residuals) <= tolerance:
        return True
    else:
        return False


def point_to_plane_distance(x, y, z, plane_params):
    """
    Calculate the perpendicular distance from a 3D point to a plane.

    Given the (X, Y, Z) coordinates of a 3D point and the coefficients of a plane in the form (a, b, c, d), where 'a', 'b',
    and 'c' are the components of the plane's normal vector, and 'd' is the offset of the plane, this function computes
    the perpendicular distance from the point to the plane.

    Parameters:
        x (float): The X-coordinate of the 3D point.
        y (float): The Y-coordinate of the 3D point.
        z (float): The Z-coordinate of the 3D point.
        plane_params (tuple): Coefficients of the plane in the form (a, b, c, d).

    Returns:
        float: The perpendicular distance from the point to the plane. With a directionality sign.
    """
    a, b, c, d = plane_params
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distance = (a*x + b*y + c*z + d) / denominator
    return distance


def project_points_onto_plane(points_dict, plane_params):
    """
    Project 3D points onto a plane.

    Given a dictionary of 3D points and the coefficients of a fitted plane, this function projects each 3D point onto the
    plane, yielding the corresponding 3D coordinates of the points in the plane. Additionally, it calculates the
    perpendicular distances from each point to the plane and stores them as 'planar_offset' in the output dictionary.

    Parameters:
        points_dict (dict): Dictionary containing point data with names as keys and (X, Y, Z) coordinates as values.
        plane_params (tuple): Coefficients of the fitted plane in the form (a, b, c, d), where 'a', 'b', and 'c' are the
                              normal vector components, and 'd' is the offset of the plane.

    Returns:
        dict: Dictionary containing projected points with names as keys and (X, Y, Z) coordinates as values, along with
              'planar_offset' indicating the perpendicular distance of each point from the fitted plane.
    """
    a, b, c, d = plane_params
    points_projected = {}
    print('I got to projecting points on a plane')
    for point_name, point_data in points_dict.items():
        x = point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit'])
        y = point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit'])
        z = point_data['Z'] * coordinate_unit_to_mm(point_data['coordinate_unit'])

        # Project the point onto the plane
        distance = point_to_plane_distance(x, y, z, plane_params)
        x_proj = x - distance * a
        y_proj = y - distance * b
        z_proj = z - distance * c
        
        # Convert back to original units and store in the points_projected dictionary
        points_projected[point_name] = {
            'X': x_proj,
            'Y': y_proj,
            'Z': z_proj,
            'coordinate_unit': 'mm'
#            'planar_offset': distance / coordinate_unit_to_mm(point_data['coordinate_unit'])
        }

    return points_projected


def rotate_to_xy_plane(points_dict, plane_params):
    """
    Rotate the points and the plane coefficients to align with the XY plane.

    Given a dictionary of 3D points and the coefficients of a fitted plane, this function first projects each 3D point
    onto the plane, yielding the corresponding 3D coordinates of the points in the plane. It then calculates the rotation
    matrix that aligns the plane's normal vector with the Z-axis. It applies this rotation to the plane coefficients and
    the 3D points in the plane, resulting in new points that are aligned with the XY plane. Additionally, it calculates
    the perpendicular distances from each point to the rotated plane and stores them as 'planar_offset' in the output
    dictionary.

    Parameters:
        points_dict (dict): Dictionary containing point data with names as keys and (X, Y, Z) coordinates as values.
                            Each point_data should also contain 'coordinate_unit' key specifying the unit of coordinates.
        plane_params (tuple): Coefficients of the fitted plane in the form (a, b, c, d), where 'a', 'b', and 'c' are the
                              normal vector components, and 'd' is the offset of the plane.

    Returns:
        dict: Dictionary containing rotated points with names as keys and (X, Y, Z) coordinates as values, along with
              'planar_offset' indicating the perpendicular distance of each point from the rotated plane.
    """
    a, b, c, d = plane_params

    # Calculate the normal vector of the plane in the original coordinate system
    norm = np.linalg.norm([a, b, c])
    print('I got to this step 1')
    normal_vector = np.array([a/norm, b/norm, c/norm])

    # Calculate the rotation matrix to align the plane's normal vector with the Z-axis
    v = np.array([0, 0, 1])
    R = rotation_matrix_from_vectors(normal_vector, v)
    print(R)
    R_inv = rotation_matrix_from_vectors(v,normal_vector)
    print(R_inv)
    distance = point_to_plane_distance(0, 0, 0, plane_params)
    # Project the points onto the plane
    points_transformed = {}
    for point_name, point_data in points_dict.items():
        x = point_data['X'] * coordinate_unit_to_mm(point_data['coordinate_unit'])
        y = point_data['Y'] * coordinate_unit_to_mm(point_data['coordinate_unit'])
        z = point_data['Z'] * coordinate_unit_to_mm(point_data['coordinate_unit'])


        x_trans = x + distance * a
        y_trans = y + distance * b
        z_trans = z + distance * c

        # Rotate the projected point
        point_transformed = np.dot(R, np.array([x_trans, y_trans, z_trans]))
#        point_transformed = np.dot(R, np.array([x, y, z]))


        # Convert back to original units and store in the points_transformed dictionary
        points_transformed[point_name] = {
            'X': point_transformed[0],
            'Y': point_transformed[1],
            'Z': point_transformed[2],
            'coordinate_unit': 'mm'
        }

    return points_transformed, R_inv


def reverse_from_XYplane_to_original(point_rotated, plane_params,R_inv):
    a, b, c, d = plane_params
    
    # Convert 2D coordinates back to 3D
    X, Y, Z = point_rotated[0], point_rotated[1], 0

    distance = point_to_plane_distance(0, 0, 0, plane_params)
    
    print(plane_params)
    print(X,Y,Z)
    # Use the inverse of the rotation matrix to transform point back to the original 3D coordinate system
#    R_inv = np.transpose(R)
    #point_original = np.dot(np.array([X, Y, Z]), R_inv)#multiplication order looks like bug.
    point_rotated = np.dot(R_inv,np.array([X, Y, Z]))
    x_orig = point_rotated[0] - distance * a
    y_orig = point_rotated[1] - distance * b
    z_orig = point_rotated[2] - distance * c   
    
    point_original = (x_orig, y_orig, z_orig)

    print(point_original)

    return point_original


def point_distance_3D(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def rotation_matrix_from_vectors(v1, v2):
    print('I got to rotation matrix calculation')
    v1 = v1/np.linalg.norm(v1) # normalized first vector
    v2 = v2/np.linalg.norm(v2) # normalized second vector
    
    axis = np.cross(v1, v2) # axis of rotation is being established
    axis = axis/np.linalg.norm(axis) # axis is being directionalized
    
    angle = np.arccos(np.dot(v1, v2)) # angle between the two vectors
    
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3,3) + np.sin(angle) * K + (1-np.cos(angle)) *  np.matmul(K,K)
    print(R, np.linalg.det(R))
    
    return R


def plot_points_rotated_2d(points_rotated):
    """
    Plot the rotated points in 2D (after aligning with the XY plane).

    Parameters:
        points_rotated (dict): Dictionary containing rotated points 
        with names as keys and (X, Y, Z) coordinates as values.

    Returns:
        None
    """
    # Extract X, Y, and point names from the dictionary
    X = [point_data['X'] for point_data in points_rotated.values()]
    Y = [point_data['Y'] for point_data in points_rotated.values()]
    point_names = list(points_rotated.keys())

    # Create a 2D plot
    plt.scatter(X, Y, c='b', marker='o')

    # Annotate each point with its name
    for i, name in enumerate(point_names):
        plt.annotate(name, (X[i], Y[i]), textcoords="offset points", xytext=(5,5), ha='center')

    # Set axis labels and title
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rotated Points in 2D")

    # Show the plot
    plt.show()


def point_distance(point_data1, point_data2):
    x1, y1, z1 = point_data1['X'], point_data1['Y'], point_data1.get('Z')
    x2, y2, z2 = point_data2['X'], point_data2['Y'], point_data2.get('Z')

    if z1 is None or z2 is None:
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    else:
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def compare_distances(dict1, dict2, tolerance, num_pairs='all'):
    '''It does all if the argument is not there!'''
    point_names_dict1 = set(dict1.keys())
    point_names_dict2 = set(dict2.keys())

    common_point_names = point_names_dict1.intersection(point_names_dict2)

    if num_pairs == 'all':
        point_pairs = list(combinations(common_point_names, 2))
    else:
        num_pairs = min(num_pairs, len(common_point_names) * (len(common_point_names) - 1) // 2)
        point_pairs = random.sample(list(combinations(common_point_names, 2)), num_pairs)

    total_possible_pairs = len(common_point_names) * (len(common_point_names) - 1) // 2
    total_pairs_tested = len(point_pairs)
    out_of_spec_pairs = 0
    discrepancies = {}

    for point_name1, point_name2 in point_pairs:
        distance_dict1 = point_distance(dict1[point_name1], dict1[point_name2])
        distance_dict2 = point_distance(dict2[point_name1], dict2[point_name2])

        # Convert to millimeters based on unit specs or assume mm if not provided
        unit_dict1 = dict1[point_name1].get('coordinate_unit', 'mm')
        unit_dict2 = dict2[point_name1].get('coordinate_unit', 'mm')

        distance_dict1 *= coordinate_unit_to_mm(unit_dict1)
        distance_dict2 *= coordinate_unit_to_mm(unit_dict2)

        discrepancy = abs(distance_dict1 - distance_dict2)

        if discrepancy > tolerance:
            sorted_pair = tuple(sorted([point_name1, point_name2]))
            discrepancies[sorted_pair] = discrepancy

    out_of_spec_pairs = len(discrepancies)

#    print(f"Testing {total_pairs_tested} point pairs out of {total_possible_pairs} possible pairs.")
#    print(f"Found {out_of_spec_pairs} pairs out of spec.")

    if len(discrepancies) == 0:
#        print("All tested point pairs are within the tolerance.")
        return True
    else:
        print(f"Maximum discrepancy is {max(discrepancies.values())},\nminimum discrepancy is {max(discrepancies.values())}.")
        for (point_name1, point_name2), discrepancy in discrepancies.items():
            print(f"Point pair '{point_name1}' and '{point_name2}' has a discrepancy of {discrepancy:.4f}{dict2[point_name1]['coordinate_unit']}.")

        return discrepancies


def plot_cartesian_3d(points_dict, plane_params=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract point coordinates for setting plot limits
    x_coords = [point_info['X'] for point_info in points_dict.values()]
    y_coords = [point_info['Y'] for point_info in points_dict.values()]
    z_coords = [point_info['Z'] for point_info in points_dict.values()]

    # Calculate plot limits
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)

    for point_name, point_info in points_dict.items():
        x = point_info['X']
        y = point_info['Y']
        z = point_info['Z']

        ax.scatter(x, y, z)
        ax.text(x, y, z, point_name, fontsize=10, ha='right')

    if plane_params:
        a, b, c, d = plane_params
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),  
                             np.linspace(y_min, y_max, 10))
        zz = (-a * xx - b * yy - d) / c
        ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show(block=True)  # Use block=True to open in a separate window


def plot_spherical_3d(points_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point_name, point_info in points_dict.items():
        r = point_info['d']
        theta = np.radians(point_info['Hz'])
        phi = np.radians(point_info['V'])

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        ax.scatter(x, y, z)
        ax.text(x, y, z, point_name, fontsize=10, ha='right')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show(block=True)  # Use block=True to open in separate window


def merge_circle_offsets(dict1, dict2):
    merged_dict = {}

    for point_name in dict1.keys():
        if point_name in dict2:
            merged_dict[point_name] = {
                'planar_offset': dict1[point_name],
                'radial_offset': dict2[point_name]
            }

    return merged_dict


def fit_circle_3D(data_tuple, output_unit, point_transform_check_tolerance,
                  log_statistics=False, print_statistics=False):
    data_dict, file_path = data_tuple
    circle_name = os.path.splitext(os.path.basename(file_path))[0]

    # see readin function's docstring.
    plane_params, plane_statistics_dict = fit_plane(data_tuple,
                                                    output_unit,
                                                    log_statistics)

    points_projected = project_points_onto_plane(data_dict, plane_params)
    points_transformed, Rot_matrix = rotate_to_xy_plane(points_projected,
                                                        plane_params)


    if not compare_distances(data_dict, points_transformed, point_transform_check_tolerance, 'all'):
        print(f"During the {circle_name} circle fitting process, there has "
              f"been a difference in point-to-point distances between pre-"
              f" and post-transformation of tested pairs that exceeded "
              f"{point_transform_check_tolerance}mm .\nPlease review"
              f" the detailed statistics and take appropriate "
              f"actions. The fit has been performed anyway")
    circle_params_2d = fit_circle_2d((points_transformed,file_path), output_unit)
    
    circle_center_vector = reverse_from_XYplane_to_original(
                                                    circle_params_2d['center'], 
                                                    plane_params, Rot_matrix)

    out_dict = {'center': tuple(circle_center_vector), 
                'radius': circle_params_2d['radius'],
                'circle_normal_vector': plane_params[:3],
                'circle_name': circle_name,
                'offsets': merge_circle_offsets(
                                    plane_statistics_dict['planer_offsets'],
                                    circle_params_2d['point_radial_offsets']),
                'plane_statistics': plane_statistics_dict['plane_statistics'],
                'point_projected': points_projected,
                'circle_statistics': circle_params_2d['statistics']}

    return out_dict