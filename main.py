# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:40:21 2023

@author: jbarker
"""

import functions as fc
import config as cg


#keyence3d = fc.fit_circle_3d_not_general_enough(fc.read_data_from_file(cg.keyence_path), cg.output_units)
#keyence2d = fc.fit_circle_2d(fc.read_data_from_file(cg.keyence2d_path), cg.output_units, cg.logfile_path)
LT_out = fc.fit_circle_3d_not_general_enough(fc.read_data_from_file(cg.LT_path), cg.output_units_LT)
#plane_coefficients = fc.fit_plane(fc.read_data_from_file(cg.LT_path), cg.output_units_LT, cg.log_statistics)

data_dict, file_path = fc.read_data_from_file(cg.planeANDcircle_test_points_path)
data_dictLT, file_pathLT = fc.read_data_from_file(cg.LT_path)
testing_plane_coefficients = fc.fit_plane(fc.read_data_from_file(cg.planeANDcircle_test_points_path), cg.output_units, cg.log_statistics)
test_plane1 = fc.fit_plane(fc.read_data_from_file(cg.test_plane1_path), cg.output_units, cg.log_statistics)
#testing_circle_coefficients = fc.fit_circle_3d(fc.read_data_from_file(cg.planeANDcircle_test_points_path), cg.output_units_LT)
TwoD_points = fc.rotate_to_xy_plane(data_dict, testing_plane_coefficients)
#fc.plot_points_rotated_2d(TwoD_points)
#fc.compare_distances(TwoD_points, data_dict, 1e-2, 'all')

fc.plot_cartesian_3d(data_dictLT)
fc.plot_spherical_3d(data_dictLT)