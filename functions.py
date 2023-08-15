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
# from mpl_toolkits.mplot3d import Axes3D
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


def generate_noisy_ellipse_points(a, b, center_x, center_y, num_points,
                                  std_dev):
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


def generate_ellipse_points(center, semi_major_axis, semi_minor_axis,
                            num_points):
    """Generate points along an ellipse.

    Parameters
    ----------
    center : tuple of float
        Center coordinates of the ellipse (x, y).
    semi_major_axis : float
        Length of the semi-major axis.
    semi_minor_axis : float
        Length of the semi-minor axis.
    num_points : int
        Number of points to generate on the ellipse.

    Returns
    -------
    x_points : numpy.ndarray
        Array of x-coordinates of generated points.
    y_points : numpy.ndarray
        Array of y-coordinates of generated points.

    Example
    -------
    Center = (3, 4)
    Semi_major_axis = 5
    Semi_minor_axis = 3
    Num_points = 100
    x_points, y_points = generate_ellipse_points(Center, Semi_major_axis,
                                                 Semi_minor_axis, Num_points)
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    x_points = center[0] + semi_major_axis * cos_angles
    y_points = center[1] + semi_minor_axis * sin_angles

    return x_points, y_points


def read_data_from_file(file_path):
    """
    Read data from a file and return it as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the file containing the data.

    Returns
    -------
    dict
        A dictionary containing the parsed data with PointIDs as keys and
        associated values including azimuth, zenith angle, distance, Cartesian
        coordinates, and units.

    Raises
    ------
    ValueError
        If duplicate PointIDs are found in the data or an invalid data format
        is specified in the header.

    Example
    -------
    file_path = "data.txt"
    data_dict, file_path = read_data_from_file(file_path)
    """
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
                    x, y, z, coordinate_unit = spherical_to_cartesian_unit(
                        distance, d_unit, angle_unit, azimuth, zenith_angle)

                elif data_format == 'cartesian' or \
                        data_format == 'cartesian2d':

                    x = float(line[1].replace(',', '.'))
                    y = float(line[2].replace(',', '.'))
                    z = None if len(line) < 4 else float(line[3].replace(
                        ',', '.'))  # Z for Cartesian 3D, None for 2D
                # Check for duplicate PointIDs
                if PointID in data_dict:
                    raise ValueError(f"Duplicate PointID '{PointID}'"
                                     f"found in line {line_number}.")

                # Store data in the dictionary
                data_dict[PointID] = {
                    'Hz': azimuth if data_format == 'spherical' else None,
                    'V': zenith_angle if data_format == 'spherical' else None,
                    'd': distance if data_format == 'spherical' else None,
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'angle_unit': angle_unit if data_format == 'spherical'
                    else None,  # belongs to previous line, just was too long
                    'd_unit': d_unit if data_format == 'spherical' else None,
                    'coordinate_unit': coordinate_unit,
                }

    return data_dict, file_path


def distance_to_mm(unit):
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
    >>> distance_to_mm("mm")
    1.0
    >>> distance_to_mm("cm")
    10.0
    >>> distance_to_mm("m")
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

def get_angle_scale(output_units):
    """
    Calculate the scaler to convert angles to rads based on the output_units.

    Parameters
    ----------
    output_units : dict
        A dictionary containing units for different quantities.
        Example: {"angles": "deg"} for degrees.

    Returns
    -------
    float
        The scaling factor to convert angles to radians.
    """
    if output_units["angles"] == "gon":
        return np.pi / 200.0  # Convert gon to radians
    elif output_units["angles"] == "rad":
        return 1.0
    elif output_units["angles"] == "mrad":
        return 0.001
    elif output_units["angles"] == "deg":
        return np.pi / 180.0  # Convert degrees to radians
    else:
        raise ValueError(f"Invalid angle unit '{output_units['angles']}' "
                         f"specified.")


def get_angle_scale_unit(unit):
    if unit == "gon":
        return 400.0
    elif unit == "rad":
        return 1.0
    elif unit == "mrad":
        return 0.001
    elif unit == "deg":
        return 180.0 / np.pi
    else:
        raise ValueError("Invalid angle unit specified in the header.")

def make_residual_stats(residuals: Union[np.ndarray, list, tuple]):
    """
    Calculate various statistical measures from a set of residuals.

    Parameters
    ----------
    residuals : numpy.ndarray or list or tuple
        The residuals to compute statistics for.

    Returns
    -------
    dict
        A dictionary containing the following statistical measures:
        - "Standard Deviation": Standard deviation of the residuals.
        - "Maximum |Residual|": Maximum absolute value of the residuals.
        - "Minimum |Residual|": Minimum absolute value of the residuals.
        - "RMS": Root Mean Square (RMS) of the residuals.
        - "Mean": Mean value of the residuals.
        - "Median": Median value of the residuals.
    """
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
    """
    Fit a circle to 2D data points using least squares optimization.

    Parameters
    ----------
    data_tuple : tuple
        A tuple containing a dictionary of data points and the file path
        associated with the data.
    output_units : dict
        A dictionary specifying the desired output units for the circle's
        parameters.
    log_file_path : str, optional
        Path to a log file for recording fit details (default is None).

    Returns
    -------
    dict or None
        A dictionary containing information about the fitted circle if
        successful:
        - "name": Name of the fitted circle.
        - "center": Tuple containing (center_x, center_y) coordinates of the
        circle's center.
        - "radius": Radius of the fitted circle.
        - "statistics": Dictionary containing statistics about the radial
        offsets.
        - "point_radial_offsets": Dictionary containing radial offsets for
        each data point.
        Returns None if the circle fitting fails.

    Note
    ----
    This function fits a circle in 2D using least squares optimization. It
    extracts data from the input dictionary,
    fits a circle, scales the output based on the specified output units,
    calculates radial offsets,
    and provides various statistics about the fitting results.
    """
    data_dict, file_path = data_tuple
    circle_name = os.path.splitext(os.path.basename(file_path))[0]
    # Extract data from the dictionary
    x, y = [], []

    # Create a list to store point names
    point_names = []

    for point_name, point_data in data_dict.items():
        # Parse data in Cartesian format
        x.append(point_data['X'] * distance_to_mm(point_data[
            'coordinate_unit']))
        y.append(point_data['Y'] * distance_to_mm(point_data[
            'coordinate_unit']))

        # Store the point name
        point_names.append(point_name)

    # Fit a circle in 2D using least squares optimization
    initial_guess = np.array([np.mean(x), np.mean(y),
                              np.mean(np.sqrt((x - np.mean(x))**2 +
                                              (y - np.mean(y))**2))])
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
    center_x, center_y, radius = center_x * result_scale, center_y * \
        result_scale, radius * result_scale

    # Calculate radial offsets
    radial_offsets = np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius

    # Create a dictionary to store radial offsets for each point
    point_radial_offsets = {point_name: offset for point_name,
                            offset in zip(point_names, radial_offsets)}

    # Prepare statistics if requested
    statistics = make_residual_stats(radial_offsets)

    if log_file_path:
        write_circle_fit_log(log_file_path, file_path, data_dict,
                             {"center_x": center_x, "center_y": center_y},
                             radius, output_units, statistics,
                             log_statistics=False)

    return {"name": circle_name, "center": (center_x, center_y),
            "radius": radius, "statistics": statistics,
            "point_radial_offsets": point_radial_offsets}


def circle_residuals_2d(params, x, y):
    """
    Calculate the residuals for 2D circle fitting.

    Parameters
    ----------
    params : numpy.ndarray
        Array containing the parameters of the circle: center_x, center_y, and
        radius.
    x : numpy.ndarray
        Array of x-coordinates of data points.
    y : numpy.ndarray
        Array of y-coordinates of data points.

    Returns
    -------
    numpy.ndarray
        Array of residuals representing the difference between the distances
        of data points from the circle's circumference and the circle's radius.

    Note
    ----
    This function calculates the residuals for 2D circle fitting. It takes the
    center coordinates (center_x, center_y) and radius of the circle as
    parameters, along with arrays of x and y coordinates of data points. The
    residuals are the differences between the distances of data points from the
    circle's circumference and the circle's radius.
    """
    center_x, center_y, radius = params
    return np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius


def get_distance_scale(output_units):
    """
    Get the scaling factor for distance units.

    Parameters
    ----------
    output_units : dict
        Dictionary containing the desired output units, including 'distances'.

    Returns
    -------
    float
        Scaling factor for converting distances from the specified
        output units to millimeters.

    Raises
    ------
    ValueError
        If an invalid distance unit is specified in the 'distances'
        field of the output_units dictionary.

    Note
    ----
    This function calculates and returns the scaling factor to convert
    distances from the specified output units to millimeters. The valid
    distance units are "m" (meters), "cm" (centimeters), "mm"
    (millimeters), and "um" (micrometers). If an invalid distance unit
    is specified, a ValueError is raised.
    """
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
    """
    Fit a plane through 3D data points and calculate residual offsets.

    Parameters
    ----------
    data_tuple : tuple
        Tuple containing the data dictionary and the file path.
    output_units : dict
        Dictionary specifying the desired output units.
    log_statistics : str, optional
        Whether to log statistics, by default 'False'.

    Returns
    -------
    tuple
        Tuple containing the plane parameters and a dictionary with
        plane offsets and statistics.

    Raises
    ------
    ValueError
        If the input data does not contain sufficient 3D points for
        plane fitting.

    Note
    ----
    This function fits a plane through 3D data points and calculates the
    residual offsets of each point from the fitted plane. The input data
    is expected to be in the form of a data dictionary containing 'X', 'Y',
    and 'Z' coordinates. The function returns the plane parameters as
    (a, b, c, d) coefficients of the plane equation: ax + by + cz + d = 0.
    Additionally, it returns a dictionary containing point names as keys
    and their residual offsets from the fitted plane as values, along with
    statistics calculated from the residuals.
    """
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
            x.append(point_data['X'] * distance_to_mm(
                point_data['coordinate_unit']))
            y.append(point_data['Y'] * distance_to_mm(
                point_data['coordinate_unit']))
            z.append(point_data['Z'] * distance_to_mm(
                point_data['coordinate_unit']))

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
    angles = get_plane_angles(normal, output_units)

    # Calculate residuals
    plane_params = (a, b, c, d)
    residual_dict = {point_name: point_to_plane_distance(X, Y, Z, plane_params)
                     for point_name, X, Y, Z in zip(list(
                             data_dict.keys()), x, y, z)
                     }

    # Prepare statistics if requested
    statistics = make_residual_stats(np.array(list(residual_dict.values())))

    return plane_params, {'planer_offsets': residual_dict,
                          'plane_statistics': statistics,
                          'angles form axis': {'Rx': angles[0],
                                               'Ry': angles[1],
                                               'Rz': angles[2]
                                               }
                          }


def get_plane_angles(normal_vector, output_units):
    """
    Calculate the angles of a plane's normal vector to CS' axes.

    Parameters
    ----------
    normal_vector : np.ndarray
        Normal vector of the plane.
    output_units : dict
        Dictionary specifying the desired output units.

    Returns
    -------
    tuple
        Tuple containing the angles in radians or converted output units.

    Note
    ----
    This function calculates the angles of a plane's normal vector
    with respect to the coordinate axes. The normal vector should be
    provided as a NumPy ndarray. The function returns a tuple containing
    the angles calculated for the X, Y, and Z axes. The angles are
    expressed in radians by default or can be converted to the desired
    output units specified in the `output_units` dictionary.
    """
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    Rx = np.arccos(normal_vector[0])
    Ry = np.arccos(normal_vector[1])
    Rz = np.arccos(normal_vector[2])
    # Convert angles to the desired output units
    angle_scale = 1/get_angle_scale(output_units)
    Rx *= angle_scale
    Ry *= angle_scale
    Rz *= angle_scale
    return Rx, Ry, Rz


def write_circle_fit_log(log_file_path, file_path, data_dict, center, radius,
                         output_units, statistics, log_statistics=False):
    """
    Write fitting results and statistics of a circle fit to a log file.

    Parameters
    ----------
    log_file_path : str
        Path to the log file.
    file_path : str
        Path to the source file.
    data_dict : dict
        Dictionary containing the data points used for circle fitting.
    center : dict
        Dictionary containing the center coordinates of the fitted circle.
    radius : float
        Radius of the fitted circle.
    output_units : dict
        Dictionary specifying the units used for output.
    statistics : dict
        Dictionary containing statistics related to the circle fit.
    log_statistics : bool, optional
        Flag indicating whether to log the statistics, by default False.

    Returns
    -------
    bool
        Returns True if the log file was successfully written.

    Note
    ----
    This function writes the results of a circle fitting operation and
    associated statistics to a log file. It receives parameters including
    the path to the log file, the source file, data points, circle center,
    radius, units, and statistics. The function also accepts a flag to
    determine whether to log the statistics or not. The log file is
    formatted with date and time information, as well as fitting details
    and optional statistics if requested.
    """
    if not log_file_path:
        return

    with open(log_file_path, 'a+') as log_file:
        circle_name = os.path.basename(file_path)
        # Write header with date and time of calculation
        log_file.write("_" * 100)
        log_file.write("\nCircle {} Fitting Results:\n".format(circle_name))
        log_file.write("Calculation Date: {}\n".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        log_file.write("Source File: {}\n".format(file_path))
        log_file.write("Units: {}\n".format(output_units["distances"]))
        log_file.write("\n")

        # Write circle fitting results
        if "center_z" in center:
            log_file.write("Center: {},{},{}\n".format(center["center_x"],
                                                       center["center_y"],
                                                       center["center_z"]))
        else:
            log_file.write("Center: {},{}\n".format(center["center_x"],
                                                    center["center_y"]))

        log_file.write("Radius: {}\n".format(radius))
        log_file.write("\n")

        # Write statistics if available and log_statistics is True
        if log_statistics:
            log_file.write("Best-fit Statistics:\n")
            for key, value in statistics.items():
                log_file.write("{}: {}\n".format(key, value))
            log_file.write("\n")
    return


def check_plane_projection(points, center, normal_vector, tolerance):
    """
    Check if the points lie on fitted plane or need to be projected onto it.

    Parameters
    ----------
    points : list of tuples
        List of (x, y, z) points to be checked.
    center : tuple
        Center of the fitted plane in (x, y, z) coordinates.
    normal_vector : tuple
        Normal vector of the fitted plane in (a, b, c) form.
    tolerance : float
        Maximum allowable residual to consider points lying on the plane.

    Returns
    -------
    bool
        True if points lie on the plane, False if points need to be projected
        onto the plane.

    Note
    ----
    This function checks whether a list of given points lies on the fitted
    plane, based on their residuals from the plane. The parameters include the
    points to be checked, the center and normal vector of the fitted plane, and
    a tolerance value for residuals. The function calculates the residuals of
    each point from the plane and checks if the maximum residual is within the
    specified tolerance. If the maximum residual is below the tolerance, the
    function returns True, indicating that the points lie on the plane.
    Otherwise, it returns False, indicating that the points need to be
    projected onto the plane.
    """
    # Calculate the residuals of the points from the fitted circle
    residuals = [np.dot(np.array(point) - np.array(center),
                        np.array(normal_vector)) for point in points]

    # Check if the maximum residual is within the tolerance
    if max(residuals) <= tolerance:
        return True
    else:
        return False


def point_to_plane_distance(x, y, z, plane_params):
    """
    Calculate perpendicular distance from a 3D point to a plane.

    Given (X, Y, Z) coords of a 3D point and plane coeffs (a, b, c, d), where
    a, b, c are plane's normal vector components, d is the plane offset,
    this function computes perpendicular distance from point to the plane.

    Parameters
    ----------
    x : float
        X-coordinate of the 3D point.
    y : float
        Y-coordinate of the 3D point.
    z : float
        Z-coordinate of the 3D point.
    plane_params : tuple
        Coefficients of the plane as (a, b, c, d).

    Returns
    -------
    float
       Perpendicular distance from the point to the plane, with directionality.
    """
    a, b, c, d = plane_params
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distance = (a*x + b*y + c*z + d) / denominator
    return distance


def project_points_onto_plane(points_dict, plane_params):
    """
    Project 3D points onto a plane.

    Given a dictionary of 3D points and fitted plane coefficients, this
    function projects each 3D point onto the plane, resulting in corresponding
    3D coordinates on the plane. It also computes perpendicular distances from
    each point to the plane and stores them as 'planar_offset' in the output
    dictionary.

    Parameters
    ----------
    points_dict : dict
        Dictionary containing point data with names as keys and (X, Y, Z)
        coordinates as values.
    plane_params : tuple
        Coefficients of the fitted plane as (a, b, c, d), where 'a', 'b', 'c'
        are normal vector components, and 'd' is the plane's offset.

    Returns
    -------
    dict
        Dictionary with projected points (names as keys, (X, Y, Z) coordinates
        as values) and 'planar_offset' indicating perpendicular distance of
        each point from the fitted plane.
    """
    a, b, c, d = plane_params
    points_projected = {}

    for point_name, point_data in points_dict.items():
        x = point_data['X'] * distance_to_mm(point_data[
            'coordinate_unit'])
        y = point_data['Y'] * distance_to_mm(point_data[
            'coordinate_unit'])
        z = point_data['Z'] * distance_to_mm(point_data[
            'coordinate_unit'])

        # Project the point onto the plane
        distance = point_to_plane_distance(x, y, z, plane_params)
        x_proj = x - distance * a
        y_proj = y - distance * b
        z_proj = z - distance * c

        # Convert back to original units and store in the points_projected dict
        points_projected[point_name] = {
            'X': x_proj,
            'Y': y_proj,
            'Z': z_proj,
            'coordinate_unit': 'mm'
        }

    return points_projected


def rotate_to_xy_plane(points_dict, plane_params):
    """
    Rotate points and plane coefficients to align with the XY plane.

    Given a dictionary of 3D points and fitted plane coefficients, this
    function projects each 3D point onto the plane, aligning them with the XY
    plane. It calculates a rotation matrix to align the plane's normal vector
    with the Z-axis, applies the rotation to both the points and the plane
    coefficients, and computes the perpendicular distances from the rotated
    points to the plane. Results are stored as 'planar_offset' in the output
    dictionary.

    Parameters
    ----------
    points_dict : dict
        Dictionary containing point data with names as keys and (X, Y, Z)
        coordinates as values. Each point_data should also contain
        'coordinate_unit' key specifying the unit of coordinates.
    plane_params : tuple
        Coefficients of the fitted plane as (a, b, c, d), where 'a', 'b', 'c'
        are normal vector components, and 'd' is the plane's offset.

    Returns
    -------
    dict
        Dictionary with rotated points (names as keys, (X, Y, Z) coordinates
        as values) and 'planar_offset' indicating perpendicular distance of
        each point from the rotated plane. Also returns the inverse rotation
        matrix R_inv.
    """
    a, b, c, d = plane_params

    # Calculate the normal vector of the plane in the original CS
    norm = np.linalg.norm([a, b, c])

    normal_vector = np.array([a/norm, b/norm, c/norm])

    # Calculate the rotation matrix to align the plane's normal vector
    # with the Z-axis
    v = np.array([0, 0, 1])
    R = rotation_matrix_from_vectors(normal_vector, v)
    R_inv = rotation_matrix_from_vectors(v, normal_vector)

    distance = point_to_plane_distance(0, 0, 0, plane_params)
    # Project the points onto the plane
    points_transformed = {}
    for point_name, point_data in points_dict.items():
        x = point_data['X'] * distance_to_mm(point_data[
            'coordinate_unit'])
        y = point_data['Y'] * distance_to_mm(point_data[
            'coordinate_unit'])
        z = point_data['Z'] * distance_to_mm(point_data[
            'coordinate_unit'])

        x_trans = x + distance * a
        y_trans = y + distance * b
        z_trans = z + distance * c

        # Rotate the projected point
        point_transformed = np.dot(R, np.array([x_trans, y_trans, z_trans]))

        # Convert back to original units and store in the points_transformed
        # dictionary
        points_transformed[point_name] = {
            'X': point_transformed[0],
            'Y': point_transformed[1],
            'Z': point_transformed[2],
            'coordinate_unit': 'mm'
        }

    return points_transformed, R_inv


def reverse_from_XYplane_to_original(point_rotated, plane_params, R_inv):
    """
    Reverse the rotation from the XY plane back to the original 3D CS.

    Given a rotated 2D point, the coefficients of a fitted plane, and the
    inverse rotation matrix, this function converts the rotated 2D point back
    to the original 3D coordinate system. It undoes the previous rotation.

    Parameters
    ----------
    point_rotated : tuple
        Rotated 2D point in the form (X, Y).
    plane_params : tuple
        Coefficients of the fitted plane in the form (a, b, c, d), where 'a',
        'b', 'c' are normal vector components, and 'd' is the offset of the
        plane.
    R_inv : np.ndarray
        Inverse rotation matrix used to reverse the rotation.

    Returns
    -------
    tuple
        Original 3D point in the form (X, Y, Z), representing the coordinates
        in the original coordinate system before rotation.
    """
    a, b, c, d = plane_params

    # Convert 2D coordinates back to 3D
    X, Y, Z = point_rotated[0], point_rotated[1], 0

    distance = point_to_plane_distance(0, 0, 0, plane_params)
    """ NOTE ON MATH
    Next time when you are confused, why do we substract the "point
    distance", it is not using the point itself, it is using the D, which is
    basically a scalar of a vector from origin of the plane to the origin of
    the original coordinate system. When multiplied by a, b and c, it creates
    a vector in direction: plane -> origin (that is why it is substracted).
    """
    point_rotated = np.dot(R_inv, np.array([X, Y, Z]))
    x_orig = point_rotated[0] - distance * a
    y_orig = point_rotated[1] - distance * b
    z_orig = point_rotated[2] - distance * c

    point_original = (x_orig, y_orig, z_orig)

    return point_original


def rotation_matrix_from_vectors(v1, v2):
    """
    Calculate the rotation matrix that aligns one vector with another.

    Given two 3D vectors v1 and v2, this function calculates the rotation
    matrix that aligns v1 with v2 using Rodrigues' rotation formula. The
    resulting matrix can be used to rotate points or vectors from the reference
    frame defined by v1 to the reference frame defined by v2.

    Parameters
    ----------
    v1 : numpy.ndarray
        Original 3D vector to be rotated.
    v2 : numpy.ndarray
        Target 3D vector that v1 should align with.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix that transforms v1 to align with v2.

    Notes
    -----
    The algorithm used in this function is based on Rodrigues' rotation
    formula, as described in
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula.
    """
    v1 = v1/np.linalg.norm(v1)  # normalized first vector
    v2 = v2/np.linalg.norm(v2)  # normalized second vector

    axis = np.cross(v1, v2)  # axis of rotation is being established
    axis = axis/np.linalg.norm(axis)  # axis is being directionalized

    angle = np.arccos(np.dot(v1, v2))  # angle between the two vectors

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3, 3) + np.sin(angle) * K + (1-np.cos(angle)) * np.matmul(K, K)

    return R


def plot_points_2d(points):
    """
    Plot givenpoints in 2D.

    Given a dictionary of 3D or 2D points, this function generates a 2D scatter
    plot of the points. The points' X and Y coordinates are extracted from the
    dictionary, and each point is annotated with its name.

    Parameters
    ----------
    points_rotated : dict
        Dictionary containing points with names as keys and (X, Y, Z)
        coordinates as values.

    Returns
    -------
    None

    """
    # Extract X, Y, and point names from the dictionary
    X = [point_data['X'] for point_data in points.values()]
    Y = [point_data['Y'] for point_data in points.values()]
    point_names = list(points.keys())

    # Create a 2D plot
    plt.scatter(X, Y, c='b', marker='o')

    # Annotate each point with its name
    for i, name in enumerate(point_names):
        plt.annotate(name, (X[i], Y[i]), textcoords="offset points",
                     xytext=(5, 5), ha='center')

    # Set axis labels and title
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Points in 2D")

    # Show the plot
    plt.show()


def point_distance(point_data1, point_data2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    Given two dictionaries representing points with (X, Y, Z) coordinates, this
    function calculates the Euclidean distance between the points. If the
    points are in 2D space, the function calculates the distance in the XY
    plane. If the points have a Z-coordinate, the distance is calculated in 3D.

    Parameters
    ----------
    point_data1 : dict
        Dictionary containing the first point's coordinates (X, Y, Z).
    point_data2 : dict
        Dictionary containing the second point's coordinates (X, Y, Z).

    Returns
    -------
    float
        The Euclidean distance between the two points.

    Notes
    -----
    If the points are in 2D space (with missing Z-coordinates), the function
    calculates the distance in the XY plane. If both points have Z-coordinates,
    the distance is calculated in 3D space.

    Examples
    --------
    >>> point1 = {'X': 1, 'Y': 2, 'Z': 3}
    >>> point2 = {'X': 4, 'Y': 5, 'Z': 6}
    >>> distance = point_distance(point1, point2)
    >>> print(distance)
    5.196152422706632
    """
    x1, y1, z1 = point_data1['X'], point_data1['Y'], point_data1.get('Z')
    x2, y2, z2 = point_data2['X'], point_data2['Y'], point_data2.get('Z')

    if z1 is None or z2 is None:
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    else:
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def compare_distances(dict1, dict2, tolerance, num_pairs='all'):
    """
    Compare distances between point pairs and check for discrepancies.

    Given two dictionaries of 3D points, this function compares the distances
    between corresponding pairs of points. It calculates the discrepancies in
    distances between the points from both dictionaries, considering a
    specified tolerance. The function returns a dictionary of point pairs with
    discrepancies that exceed the tolerance.

    Parameters
    ----------
    dict1 : dict
        Dictionary containing point data with names as keys and (X, Y, Z)
        coordinates as values.
    dict2 : dict
        Another dictionary with the same structure as 'dict1' containing point
        data.
    tolerance : float
        Maximum allowable discrepancy between distances to consider as within
        tolerance.
    num_pairs : int or str, optional
        Number of point pairs to compare. If 'all', compare all possible pairs.
        Default is 'all'.

    Returns
    -------
    bool or dict
        If all tested point pairs are within the tolerance, returns True.
        Otherwise, returns a dictionary of point pairs and their discrepancies
        that exceed the specified tolerance.

    Notes
    -----
    - The 'coordinate_unit' key in the dictionaries specifies the unit of
      coordinates.
    - The function compares distances between corresponding points in both
      dictionaries.
    - Point pairs with discrepancies exceeding the tolerance are considered
      out of spec.

    Examples
    --------
    >>> point_data1 = {'A': {'X': 1, 'Y': 2, 'Z': 3}, 'B': {'X': 4, 'Y': 5,
                                                            'Z': 6}}
    >>> point_data2 = {'A': {'X': 1, 'Y': 2, 'Z': 3}, 'B': {'X': 4, 'Y': 5,
                                                            'Z': 6.5}}
    >>> tolerance = 0.1
    >>> result = compare_distances(point_data1, point_data2, tolerance)
    >>> if result is True:
    ...     print("All point pairs are within tolerance.")
    ... else:
    ...     print("Point pairs with discrepancies:")
    ...     for pair, discrepancy in result.items():
    ...         print(f"{pair}: {discrepancy:.4f} mm")
    """
    point_names_dict1 = set(dict1.keys())
    point_names_dict2 = set(dict2.keys())

    common_point_names = point_names_dict1.intersection(point_names_dict2)

    if num_pairs == 'all':
        point_pairs = list(combinations(common_point_names, 2))
    else:
        num_pairs = min(num_pairs, len(common_point_names) * (len(
            common_point_names) - 1) // 2)
        point_pairs = random.sample(list(combinations(
            common_point_names, 2)), num_pairs)

    total_possible_pairs = len(common_point_names) * (len(
        common_point_names) - 1) // 2
    total_pairs_tested = len(point_pairs)
    out_of_spec_pairs = 0
    discrepancies = {}

    for point_name1, point_name2 in point_pairs:
        distance_dict1 = point_distance(dict1[point_name1], dict1[point_name2])
        distance_dict2 = point_distance(dict2[point_name1], dict2[point_name2])

        # Convert to millimeters based on unit specs or assume mm if not there
        unit_dict1 = dict1[point_name1].get('coordinate_unit', 'mm')
        unit_dict2 = dict2[point_name1].get('coordinate_unit', 'mm')

        distance_dict1 *= distance_to_mm(unit_dict1)
        distance_dict2 *= distance_to_mm(unit_dict2)

        discrepancy = abs(distance_dict1 - distance_dict2)

        if discrepancy > tolerance:
            sorted_pair = tuple(sorted([point_name1, point_name2]))
            discrepancies[sorted_pair] = discrepancy

    out_of_spec_pairs = len(discrepancies)

#    print(f"Testing {total_pairs_tested} point pairs out of "
#          f"{total_possible_pairs} possible pairs.")
#    print(f"Found {out_of_spec_pairs} pairs out of spec.")

    if len(discrepancies) == 0:
        # print("All tested point pairs are within the tolerance.")
        return True
    else:
        print(f"Maximum discrepancy is {max(discrepancies.values())},"
              f"\nminimum discrepancy is {max(discrepancies.values())}.")
        for (point_name1, point_name2), discrepancy in discrepancies.items():
            print(f"Point pair '{point_name1}' and '{point_name2}' has a "
                  f"discrepancy of {discrepancy:.4f}"
                  f"{dict2[point_name1]['coordinate_unit']}.")

        return discrepancies


def plot_cartesian_3d(points_dict, plane_params=None):
    """
    Plot points in a 3D Cartesian CS with optional plane visualization.

    Given a dictionary of 3D points and optional plane coefficients, this
    function creates a 3D plot of the points in a Cartesian coordinate system.
    Points are scattered in 3D space, with their names annotated near each
    point. If plane coefficients are provided, the fitted plane is also
    visualized in the plot.

    Parameters
    ----------
    points_dict : dict
        Dictionary containing point data with names as keys and (X, Y, Z)
        coordinates as values.
    plane_params : tuple or None, optional
        Coefficients of the fitted plane in the form (a, b, c, d), where 'a',
        'b', and 'c' are the normal vector components, and 'd' is the offset of
        the plane. If None, no plane visualization is added. Default is None.

    Returns
    -------
    None

    Notes
    -----
    - The 'coordinate_unit' key in the dictionary specifies the unit of
      coordinates.
    - Points are scattered in 3D space, and their names are annotated near
      each point.
    - If 'plane_params' is provided, the fitted plane is visualized as a
      surface in the plot.

    Examples
    --------
    >>> point_data = {'A': {'X': 1, 'Y': 2, 'Z': 3}, 'B': {'X': 4, 'Y': 5,
                                                           'Z': 6}}
    >>> plot_cartesian_3d(point_data)
    >>> plot_cartesian_3d(point_data, plane_params=(1, 2, 3, 0))
    """
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
    """
    Plot 3D points in a spherical coordinate system.

    Given a dictionary of 3D points defined in spherical coordinates (radius
    'd', azimuthal angle 'Hz', and polar angle 'V'), this function creates a
    3D plot of the points in a spherical coordinate system. Points are
    scattered in 3D space based on their spherical coordinates, and their names
    are annotated near each point.

    Parameters
    ----------
    points_dict : dict
        Dictionary containing point data with names as keys and spherical
        coordinates as values. Each point's data should include 'd' (radius),
        'Hz' (azimuthal angle in degrees), and 'V' (polar angle in degrees).

    Returns
    -------
    None

    Notes
    -----
    - Points are scattered in 3D space based on their spherical coordinates
      (radius, azimuthal angle, and polar angle).
    - The 'd' values represent the radial distance, 'Hz' values represent the
      azimuthal angle in degrees, and 'V' values represent the polar angle in
      degrees.
    - Points' names are annotated near each point in the plot.

    Examples
    --------
    >>> point_data = {'A': {'d': 1, 'Hz': 45, 'V': 30},
                      'B': {'d': 2, 'Hz': 60, 'V': 60}}
    >>> plot_spherical_3d(point_data)
    """
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
    """
    Merge two dictionaries containing circle offsets into a single dictionary.

    Given two dictionaries containing circle offsets, one with 'planar_offset'
    and the other with 'radial_offset', this function merges the two
    dictionaries into a single dictionary. The merged dictionary will have
    point names as keys, and for each point, it will include both
    'planar_offset' and 'radial_offset' values.

    Parameters
    ----------
    {dict1_name} : dict
        Dictionary containing 'planar_offset' values for circle points, with
        point names as keys.
    {dict2_name} : dict
        Dictionary containing 'radial_offset' values for circle points, with
        point names as keys.

    Returns
    -------
    dict
        Merged dictionary containing point names as keys and a sub-dictionary
        with 'planar_offset' and 'radial_offset' values as values.

    Raises
    ------
    ValueError
        If the dictionaries do not have the same set of point names.

    Notes
    -----
    - The function assumes that both dictionaries share common point names.
    - The merged dictionary will include a sub-dictionary for each common
      point name, with 'planar_offset' and 'radial_offset' values from
      {dict1_name} and {dict2_name}, respectively.
    """
    dict1_name = get_variable_name(dict1)
    dict2_name = get_variable_name(dict2)

    point_names_dict1 = set(dict1.keys())
    point_names_dict2 = set(dict2.keys())

    common_point_names = point_names_dict1.intersection(point_names_dict2)

    if len(common_point_names) == 0:
        raise ValueError(f"No common point names between {dict1_name} and"
                         f"{dict2_name}.")

    if len(common_point_names) != len(point_names_dict1) or len(
            common_point_names) != len(point_names_dict2):
        raise ValueError(f"Point names in {dict1_name} and {dict2_name} do"
                         f" not match.")

    merged_dict = {}

    for point_name in common_point_names:
        merged_dict[point_name] = {
            'planar_offset': dict1[point_name],
            'radial_offset': dict2[point_name]
        }

    return merged_dict


def fit_circle_3D(data_tuple, output_unit, point_transform_check_tolerance,
                  log = False, log_statistics=False):
    """
    Fit a 3D circle to a set of 3D points in space.

    Given a data tuple containing a dictionary of 3D points and the associated
    file path, this function performs a circle fitting process in 3D space. It
    begins by fitting a plane to the given points using the 'fit_plane'
    function. The points are then projected onto this fitted plane, followed by
    a rotation transformation that aligns the plane with the XY plane. The
    distances between the original and transformed points are compared using
    the 'compare_distances' function to check for any discrepancies after the
    transformation.

    Parameters
    ----------
    data_tuple (tuple): A tuple containing the dictionary of 3D points and the
                        associated file path.
    output_unit (dict): Dictionary specifying the output units for distances.
    point_transform_check_tolerance (float): Tolerance for checking differences
                                             in point-to-point distances after
                                             transformation.
    log_statistics (bool, optional): If True, log statistics will be generated
                                     and stored. Defaults to False.
    print_statistics (bool, optional): If True, statistics will be printed.
                                       Defaults to False.

    Returns
    -------
    dict: A dictionary containing the fitted circle parameters and associated
    information, including:
          - 'center': The 3D coordinates of the circle's center.
          - 'radius': The radius of the fitted circle.
          - 'circle_normal_vector': The normal vector of the fitted circle's
             plane.
          - 'circle_name': The name of the circle.
          - 'offsets': Dictionary containing planar and radial offsets.
          - 'plane_statistics': Statistics from the fitted plane.
          - 'plane_angles_parameters': Parameters related to angles of the
             fitted plane.
          - 'circle_statistics': Statistics from the fitted circle.

    Notes
    -----
    - The function performs circle fitting in the following steps:
        1. Fit a plane to the 3D points.
        2. Project points onto the fitted plane.
        3. Rotate the points to align with the XY plane.
        4. Check for differences in point-to-point distances before and after
            transformation.
        5. Fit a 2D circle to the rotated points.
    - The output dictionary contains various statistical information about the
        fitted plane and circle.
    """
    data_dict, file_path = data_tuple
    circle_name = os.path.splitext(os.path.basename(file_path))[0]

    # see readin function's docstring.
    plane_params, plane_statistics_dict = fit_plane(data_tuple,
                                                    output_unit,
                                                    log_statistics)

    points_projected = project_points_onto_plane(data_dict, plane_params)
    points_transformed, Rot_matrix = rotate_to_xy_plane(points_projected,
                                                        plane_params)

    if not compare_distances(data_dict, points_transformed,
                             point_transform_check_tolerance, 'all'):
        print(f"During the {circle_name} circle fitting process, there has "
              f"been a difference in point-to-point distances between pre-"
              f" and post-transformation of tested pairs that exceeded "
              f"{point_transform_check_tolerance}mm .\nPlease review"
              f" the detailed statistics and take appropriate "
              f"actions. The fit has been performed anyway")
    circle_params_2d = fit_circle_2d((points_transformed, file_path),
                                     output_unit)

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
                'plane_angles_parameters': plane_statistics_dict[
                    'angles form axis'],
                'circle_statistics': circle_params_2d['statistics']}
    if log and not log_statistics:
        write_3D_circle_fit_log(out_dict, output_unit, log)
    else: write_3D_circle_fit_log(out_dict, output_unit, log, log_statistics)

    return out_dict


def write_3D_circle_fit_log(results_dict, output_units, log_details,
                            log_statistics=False):
    """
    Write fitting results and statistics of a 3D circle fit to a log file.

    Parameters
    ----------
    log_file_path : str
        Path to the log file.
    results_dict : dict
        Dictionary containing the results of the 3D circle fit. The dictionary
        should include various parameters and statistics related to the fit.
    log_statistics : bool, optional
        Flag indicating whether to log the statistics, by default False.

    Returns
    -------
    bool
        Returns True if the log file was successfully written.

    Note
    ----
    This function writes the results of a 3D circle fitting operation and
    associated statistics to a log file. It receives the path to the log file
    and a dictionary containing the results of the 3D circle fit. The 
    dictionary should include parameters such as the circle's center, radius,
    normal vector, offsets, plane statistics, circle statistics, etc. The log
    file is formatted with date and time information, as well as fitting 
    details and optional statistics if requested.
    """

    log_path, log_precision = log_details
    decimal_places_distances, decimal_places_angles = calculate_decimal_places(output_units, log_precision)

    with open(log_path, 'a+') as log_file:
        circle_name = results_dict.get('circle_name', 'Unknown Circle')
        distances_unit = output_units.get('distances', 'N/A')
        angles_unit = output_units.get('angles', 'N/A')
        
        # Write header with date and time of calculation
        log_file.write("_" * 100)
        log_file.write("\n3D Circle Fitting Results:\n")
        log_file.write("Calculation Date: {}\n".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        log_file.write("Circle Name: {}\n".format(circle_name))
        log_file.write(f"Units: Distances [{distances_unit}], Angles [{angles_unit}]\n")
        log_file.write("\n")

        # Write circle fitting results with formatted numbers
        log_file.write("Center: {}\n".format(
            tuple(format(value, f".{decimal_places_distances}f")
                  for value in results_dict.get('center', 'N/A'))))
        log_file.write("Radius: {}\n".format(
            format(results_dict.get('radius', 'N/A'), 
                   f".{decimal_places_distances}f")))
        log_file.write("Circle Normal Vector: {}\n".format(
            ", ".join([f"{value:.{decimal_places_distances}f}" 
               for value in results_dict.get('circle_normal_vector', 'N/A')])))
        log_file.write("\n")

        # Write statistics if available and log_statistics is True
        if log_statistics:
            circle_statistics = results_dict.get('circle_statistics', {})
            log_file.write("Circle Fit Statistics:\n")
            for key, value in circle_statistics.items():
                if isinstance(value, float):
                    value_str = f"{value:.{decimal_places_distances}f}"
                else:
                    value_str = value
                log_file.write("{}: {}\n".format(key, value_str))
            log_file.write("\n")

            # Write offsets as a table
            offsets = results_dict.get('offsets', {})
            if offsets:
                log_file.write("Offsets:\n")
                log_file.write("Point ID Planar Offset Radial Offset\n")
                for point_id, offset_dict in offsets.items():
                    planar_offset = offset_dict.get('planar_offset', 'N/A')
                    radial_offset = offset_dict.get('radial_offset', 'N/A')
        
                    # Format and align the columns with specified decimal places
                    planar_str = \
                    f"{planar_offset:.{decimal_places_distances}f}".rjust(12)
                    radial_str = \
                    f"{radial_offset:.{decimal_places_distances}f}".rjust(12)
                    log_file.write(f"{point_id}\t{planar_str}\t{radial_str}\n")
                log_file.write("\n")
    return


def calculate_decimal_places(output_units, log_precision):
    """
    Calculate decimal places based on output units and log precision.

    This function calculates the number of decimal places required for both
    distances and angles in the context of logging fitting results. It takes
    into account the output units and the specified log precision for distances
    and angles.

    Parameters
    ----------
    output_units : dict
        Dictionary specifying the units used for output distances and angles.
    log_precision : dict
        Dictionary specifying the log precision for distances and angles.

    Returns
    -------
    tuple
        A tuple containing two integers representing the number of decimal
        places needed for distances and angles in the log.

    Notes
    -----
    - The function performs the following steps:
      1. Scales the log precision to millimeters (mm).
      2. Scales the log precision in millimeters to the units of the
         output_units.
      3. Determines the number of decimal places required for distances based
         on the scaled log precision.
      4. Scales the log precision for angles to radians.
      5. Scales the log precision in radians to the units of the output_units.
      6. Determines the number of decimal places required for angles based on
         the scaled log precision.
    """
    
    distances_precision, distances_precision_unit = log_precision['distances']
    angles_precision, angles_precision_unit = log_precision['angles']

    # Scale log_precision to mm
    if distances_precision_unit != 'mm':
        log_precision_scaled_mm = distances_precision * distance_to_mm(
            distances_precision_unit)
    else:
        log_precision_scaled_mm = distances_precision
    # print('Distance precision log:')
    # print(distances_precision, distances_precision_unit)
    # print('Distance precision Output')
    # print(output_units['distances'])
    # print(f'log_precision_scaled_mm:{log_precision_scaled_mm}')

    # Scale log_precision in mm to the units of output_units
    if output_units['distances'] != 'mm':
        log_precision_scaled_output = log_precision_scaled_mm / distance_to_mm(
            output_units['distances'])
    else:
        log_precision_scaled_output = log_precision_scaled_mm

    # print(f'log_precision_scaled_output:{log_precision_scaled_output}')

    # Determine the number of decimal places for distances
    decimal_places_distances = max(-int(np.floor(np.log10(
        log_precision_scaled_output))), 0)
    # print(f'decimal_places_distances:{decimal_places_distances}')
    
    # Scale log_precision for angles to radians
    if angles_precision_unit != 'rad':
        log_precision_scaled_rad = angles_precision * get_angle_scale_unit(
            angles_precision_unit)
    else:
        log_precision_scaled_rad = angles_precision

    # Scale log_precision in radians to the units of output_units
    if output_units['angles'] != 'rad':
        log_precision_scaled_output_angles = log_precision_scaled_rad / \
            get_angle_scale(output_units)
    else:
        log_precision_scaled_output_angles = log_precision_scaled_rad

    # Determine the number of decimal places for angles
    decimal_places_angles = int(np.floor(
        np.log10(1.0 / log_precision_scaled_output_angles)))

    return decimal_places_distances+1, decimal_places_angles+1