# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:40:21 2023

@author: jbarker
"""

import numpy as np
from scipy.optimize import least_squares
import functions as fc
import config as cg


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
    """
    print("Estimated Parameters:")
    print("Semi-Major Axis (a):", a_estimated)
    print("Semi-Minor Axis (b):", b_estimated)
    print("Center (x, y):", center_x_estimated, center_y_estimated)
    """
    
    return a, b, center_x, center_y


keyence3d = fc.fit_circle_3d(fc.read_data_from_file(cg.keyence_path), cg.output_units)
keyence2d = fc.fit_circle_2d(fc.read_data_from_file(cg.keyence2d_path), cg.output_units, cg.logfile_path)
LT = fc.fit_circle_3d(fc.read_data_from_file(cg.LT_path), cg.output_units_LT)
plane_coefficients = fc.fit_plane_3d(fc.read_data_from_file(cg.LT_path), cg.output_units_LT, cg.log_statistics)

data_dict, file_path = fc.read_data_from_file(cg.planeANDcircle_test_points_path)
testing_plane_coefficients = fc.fit_plane_3d(fc.read_data_from_file(cg.planeANDcircle_test_points_path), cg.output_units_LT, cg.log_statistics)
testing_circle_coefficients = fc.fit_circle_3d(fc.read_data_from_file(cg.planeANDcircle_test_points_path), cg.output_units_LT)

