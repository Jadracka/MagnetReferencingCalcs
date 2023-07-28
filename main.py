# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:40:21 2023

@author: jbarker
"""

import numpy as np
from scipy.optimize import least_squares


"""Generating clean points of ellipse"""

def generate_ellipse_points(center, semi_major_axis, semi_minor_axis, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    x_points = center[0] + semi_major_axis * cos_angles
    y_points = center[1] + semi_minor_axis * sin_angles

    return x_points, y_points

# Example usage:
Center = (3, 4)  # Center coordinates of the ellipse
Semi_major_axis = 5  # Length of the semi-major axis
Semi_minor_axis = 3  # Length of the semi-minor axis
Num_points = 100  # Number of points to generate on the ellipse

x_points, y_points = generate_ellipse_points(Center, Semi_major_axis, Semi_minor_axis, Num_points)

# Now x_points and y_points contain the coordinates of points lying on the ellipse without noise.

def fit_ellipse_lsm(x, y):
    def ellipse_residuals(params, x, y):
        a, b, center_x, center_y = params
        return a**2 * (x - center_x)**2 + b**2 * (y - center_y)**2 - a**2 * b**2

    # Initial guess for ellipse parameters (semi-major axis, semi-minor axis, center_x, center_y)
    initial_guess = [1.0, 1.0, np.mean(x), np.mean(y)]

    # Fit the ellipse parameters using least squares
    result = least_squares(ellipse_residuals, initial_guess, args=(x, y))

    # Extract the fitted ellipse parameters
    a, b, center_x, center_y = result.x

    return a, b, center_x, center_y

# Example usage:
    
def generate_noisy_ellipse_points(a, b, center_x, center_y, num_points, std_dev):
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

# Example usage:
a = 3.0  # Semi-major axis
b = 1.5  # Semi-minor axis
center_x = 2.0  # x-coordinate of the center
center_y = 1.0  # y-coordinate of the center
num_points = 100  # Number of points to generate
std_dev = 0.1  # Standard deviation for the noise (adjust as desired)

x_values, y_values = generate_noisy_ellipse_points(a, b, center_x, center_y, num_points, std_dev)

# Fit the noisy ellipse points and get the estimated parameters
a_estimated, b_estimated, center_x_estimated, center_y_estimated = fit_ellipse_lsm(x_values, y_values)

print("True Parameters:")
print("Semi-Major Axis (a):", a)
print("Semi-Minor Axis (b):", b)
print("Center (x, y):", center_x, center_y)
print()

print("Estimated Parameters:")
print("Semi-Major Axis (a):", a_estimated)
print("Semi-Minor Axis (b):", b_estimated)
print("Center (x, y):", center_x_estimated, center_y_estimated)
