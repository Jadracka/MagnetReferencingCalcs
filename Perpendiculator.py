# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:23:20 2021

@author: jbarker
"""

import numpy as np
import math
from angle import Angle as a
import functions as fc



# Souřadnice testovací
# Coord1 = np.array([
#     [1, 499.997, 500.003],
#     [2, 499.997, 1000.003],
#     [3, 1000.003, 999.997],
#     [4, 1000.004, 500.004]
# ])

# Coord2 = np.array([
#     [1, 357.701, 626.169],
#     [2, 134.094, 1073.383],
#     [3, 581.448, 1296.849],
#     [4, 804.914, 849.636]
# ])


def perpendiculator(To, From):
    if not isinstance(To, dict) or not isinstance(From, dict):
        raise TypeError("Both 'To' and 'From' arguments must be"
						"dictionaries")
    To_point_names = set(To.keys())
    From_point_names = set(From.keys())

    common_point_names = To_point_names.intersection(From_point_names)
    
    plane_dict = fit_plane(data_tuple, output_unit)
    plane_params = plane_dict['plane_parameters']

    points_projected = project_points_onto_plane(data_dict, plane_params)
    points_transformed, Rot_matrix = rotate_to_xy_plane(points_projected,
                                                        plane_params)
    # Těžiště místní soustavy
    Ty = np.sum(Coord2[:, 1]) / Coord2.shape[0]
    Tx = np.sum(Coord2[:, 2]) / Coord2.shape[0]

    # Těžiště hlavní soustavy
    TY = np.sum(Coord1[:, 1]) / Coord1.shape[0]
    TX = np.sum(Coord1[:, 2]) / Coord1.shape[0]

    # Výpočet redukovaných souřadnic v místní soustavě
    yT = np.tile(Ty, (Coord2.shape[0], 1))
    xT = np.tile(Tx, (Coord2.shape[0], 1))
    yred = Coord2[:, 1] - yT
    xred = Coord2[:, 2] - xT

    # Výpočet redukovaných souřadnic v hlavní soustavě
    YT = np.tile(TY, (Coord1.shape[0], 1))
    XT = np.tile(TX, (Coord1.shape[0], 1))
    Yred = Coord1[:, 1] - YT
    Xred = Coord1[:, 2] - XT

    # Výpočet transformačních koeficientù
    A = xred * Xred
    B = yred * Yred
    C = (xred**2) + (yred**2)

    lam1 = (np.sum(A) + np.sum(B)) / np.sum(C)  # koeficient lambda 1

    D = xred * Yred
    E = yred * Xred

    lam2 = (np.sum(D) - np.sum(E)) / np.sum(C)  # koeficient lambda 2

    t = [TY - (lam2 * Tx + lam1 * Ty), TX - (lam1 * Tx - lam2 * Ty)]

    q = np.sqrt(lam1**2 + lam2**2)

    om = np.arctan2(lam2, lam1)
    om_g = om * 200 / np.pi

    klic = np.array([lam1, lam2, t[1], t[0], q, om_g])

    # Souřadnice podobnostní transformací
    F = lam1 * Coord2[:, 2]
    G = lam2 * Coord2[:, 1]
    I = lam2 * Coord2[:, 2]
    J = lam1 * Coord2[:, 1]

    Yn = I + J + t[0]
    Xn = F - G + t[1]

    Sour = np.column_stack([Yn, Xn])

    vxy = Sour - Coord1[:, 1:]
    vx = vxy[:, 1]
    vy = vxy[:, 0]

    # Pokles sumy čtvercù
    # Podobnostní
    dvv = (vx**2 + vy**2) / (((Coord2.shape[0] - 1) / Coord2.shape[0]) - ((xred**2 + yred**2) / np.sum(xred**2 + yred**2)))

    return klic, dvv, vx, vy

def trn_klic_shod2(Coord1, Coord2)  :
    # Těžiště místní soustavy
    Ty = np.sum(Coord2[:, 1]) / Coord2.shape[0]
    Tx = np.sum(Coord2[:, 2]) / Coord2.shape[0]

    # Těžiště hlavní soustavy
    TY = np.sum(Coord1[:, 1]) / Coord1.shape[0]
    TX = np.sum(Coord1[:, 2]) / Coord1.shape[0]

    # Výpočet redukovaných souřadnic v místní soustavě
    yT = np.tile(Ty, (Coord2.shape[0], 1))
    xT = np.tile(Tx, (Coord2.shape[0], 1))
    yred = Coord2[:, 1] - yT
    xred = Coord2[:, 2] - xT

    # Výpočet redukovaných souřadnic v hlavní soustavě
    YT = np.tile(TY, (Coord1.shape[0], 1))
    XT = np.tile(TX, (Coord1.shape[0], 1))
    Yred = Coord1[:, 1] - YT
    Xred = Coord1[:, 2] - XT

    # Výpočet transformačních koeficientù
    A = xred * Xred
    B = yred * Yred
    C = (xred**2) + (yred**2)

    lam1 = (np.sum(A) + np.sum(B)) / np.sum(C)  # koeficient lambda 1

    D = xred * Yred
    E = yred * Xred

    lam2 = (np.sum(D) - np.sum(E)) / np.sum(C)  # koeficient lambda 2
    q = np.sqrt(lam1**2 + lam2**2)

    # Shodnostní transformace bez použití redukcí k těžišti
    lams1 = lam1 / q
    lams2 = lam2 / q

    om = np.arctan2(lams2, lams1)
    tsy = TY - (lams2 * Tx + lams1 * Ty)
    tsx = TX - (lams1 * Tx - lams2 * Ty)

    klic = np.array([lams1, lams2, tsx, tsy, om * 200 / np.pi])
    K = lams1 * Coord2[:, 1]
    L = lams2 * Coord2[:, 2]
    M = lams1 * Coord2[:, 2]
    N = lams2 * Coord2[:, 1]

    vy = L + K + tsy - Coord1[:, 1]
    vx = M - N + tsx - Coord1[:, 2]

    return klic, vx, vy

def trn_klic_shod(Coord1, Coord2):
    klic, vx, vy = trn_klic_shod2(Coord1, Coord2)
    dv0 = np.sum(vx**2 + vy**2)
    dvv = []
    for j in range(1, vx.shape[0] + 1):
        klictemp, vxtemp, vytemp = trn_klic_shod2(drop_line(Coord1, j), drop_line(Coord2, j))
        dvv.append(dv0 - np.sum(vxtemp**2 + vytemp**2))
    return klic, vx, vy, np.array(dvv)

klic, dvv, vx, vy = trn_klic_pod(Coord1, Coord2)
klics, dvvs, vxs, vys = trn_klic_shod(Coord1, Coord2)



#    Transformed_From = Transformation(x,From)
#    Trans_par = np.array([x[0], x[1], x[2], x[3],
#                          a(x[4], a.T_RAD, True).angle,
#                          a(x[5], a.T_RAD, True).angle,
 #                         a(x[6], a.T_RAD, True).angle])



def pretty_print(x):
    for i in x:
        print("{:7.2f} ".format(i), end='')
    print()
    return ()