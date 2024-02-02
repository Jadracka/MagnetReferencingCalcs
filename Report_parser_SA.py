# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:40:32 2023

@author: jbarker
"""

import config as cg
import pandas as pd

def read_SA_instrument_report(file_path):
    try:
        # Check the file extension to determine the engine
        if file_path.endswith('.xlsx'):
            engine = 'openpyxl'
        else:
            raise ValueError("Unsupported file format. Only .xlsx and .xls are supported.")

        # Read the file using the determined engine
        df = pd.read_excel(file_path, engine=engine)

        # Remove rows with NaN in the first column
        df = df.dropna(subset=[df.columns[0]])

        return df
    except Exception as e:
        print("Error:", str(e))
        return None

dict_measurements = {}


measurement_file = read_SA_instrument_report(cg.measurement_file_path)

Collection_occures = measurement_file[
    measurement_file.columns[0]].str.contains('Collection',
                                              case=True, na=False)
Collection_coords = list(Collection_occures[Collection_occures].index)

number_of_instruments = len(Collection_coords)



measurement_file2 = read_SA_instrument_report(cg.more_inst_file_path)

Collection_occures2 = measurement_file2[
    measurement_file2.columns[0]].str.contains('Collection',
                                              case=True, na=False)
Collection_coords2 = list(Collection_occures2[Collection_occures2].index)

number_of_instruments2 = len(Collection_coords2)