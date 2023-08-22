"""
Created on Thu Jul 13 09:40:21 2023

@author: jbarker
"""

import functions as fc
import config as cg


LT_out = fc.fit_circle_3D(fc.read_data_from_file(cg.LT_path),
                          cg.output_units_LT,
                          cg.point_transform_check_tolerance, cg.log,
                          cg.log_statistics)


Keyence_Dnstr = fc.fit_circle_2d(fc.read_data_from_file(cg.keyence2d_path),
                                 cg.output_units, cg.log, cg.log_statistics)
