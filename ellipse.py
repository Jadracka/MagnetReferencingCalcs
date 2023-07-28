# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:18:15 2023

@author: jbarker
"""
import numpy as np


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
b = 3.0  # Semi-minor axis
center_x = 2.0  # x-coordinate of the center
center_y = 1.0  # y-coordinate of the center
num_points = 100  # Number of points to generate
std_dev = 0.1  # Standard deviation for the noise (adjust as desired)

x_values, y_values = generate_noisy_ellipse_points(a, b, center_x, center_y, num_points, std_dev)


def fit_ellipse(x, y):
    D = np.column_stack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    try:
        E, V = np.linalg.eig(np.linalg.inv(S) @ C)
    except np.linalg.LinAlgError:
        # If the matrix is not invertible, return None (no ellipse found)
        return None

    # Select the positive eigenvalue
    real_eigs = E.real
    idx = real_eigs > 0
    a = V[:, idx][:, 0]

    # Convert coefficients to ellipse parameters
    a11, a12, a22, a1, a2, a0 = a
    center_x = (a12 * a2 - 2 * a22 * a1) / (4 * a11 * a22 - a12 ** 2)
    center_y = (a12 * a1 - 2 * a11 * a2) / (4 * a11 * a22 - a12 ** 2)

    num = 2 * (a11 * center_x ** 2 + a22 * center_y ** 2 + a1 * center_x + a2 * center_y + a0)
    denom = a11 + a22 - np.sqrt((a11 - a22) ** 2 + a12 ** 2)
    semi_major_axis = np.sqrt(abs(num / denom))
    semi_minor_axis = np.sqrt(abs(num / ((a11 + a22) + np.sqrt((a11 - a22) ** 2 + a12 ** 2))))

    orientation_rad = 0.5 * np.arctan2(a12, (a11 - a22))
    orientation_deg = np.degrees(orientation_rad)

    return (center_x, center_y, semi_major_axis, semi_minor_axis, orientation_deg)

# Example usage:
#x_values = np.array([1, 2, 3, 4, 5])
#y_values = np.array([2, 4, 1, 3, 5])
ellipse_parameters = fit_ellipse(x_values, y_values)

if ellipse_parameters is not None:
    center_x, center_y, semi_major_axis, semi_minor_axis, orientation_deg = ellipse_parameters
    print("Ellipse Center (x, y):", (center_x, center_y))
    print("Semi-Major Axis:", semi_major_axis)
    print("Semi-Minor Axis:", semi_minor_axis)
    print("Orientation (degrees):", orientation_deg)
else:
    print("No ellipse could be fit to the data.")