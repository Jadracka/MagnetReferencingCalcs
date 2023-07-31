# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:40:21 2023

@author: jbarker
"""

import numpy as np
from scipy.optimize import least_squares


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