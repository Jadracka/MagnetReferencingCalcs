"""
Created on Thu Jul 13 09:40:21 2023

@author: jbarker
"""



import functions as fc
import config as cg


# keyence3d = fc.fit_circle_3D(fc.read_data_from_file(cg.keyence_path),
#                        cg.output_units, cg.point_transform_check_tolerance)
# LT_out = fc.fit_circle_3D(fc.read_data_from_file(cg.LT_path),
#                          cg.output_units_LT,
#                          cg.point_transform_check_tolerance)
#plane_coefficients = fc.fit_plane(fc.read_data_from_file(cg.LT_path), cg.output_units_LT, cg.log_statistics)

#data_dictLT, file_pathLT = fc.read_data_from_file(cg.LT_path)
#testing_plane_coefficients, testing_plane_statistics = fc.fit_plane(fc.read_data_from_file(cg.planeANDcircle_test_points_path), cg.output_units, cg.log_statistics)
#test_plane1, test_plane1_statistics = fc.fit_plane(fc.read_data_from_file(cg.test_plane1_path), cg.output_units, cg.log_statistics)
testing_circle_coefficients = fc.fit_circle_3D(
        fc.read_data_from_file(cg.planeANDcircle_test_points_path),
        cg.output_units_LT,
        cg.point_transform_check_tolerance, cg.log, cg.log_statistics)

