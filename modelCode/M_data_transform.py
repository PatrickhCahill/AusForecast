'''
A file that will transfrom the raw data a processed form that will make the model simpler to run.
Generally this is stripping the raw AEC data down to only required rows and then combining them into a single matrix
'''

# Imports
import os, sys
import numpy as np
import pandas as pd

# Functions

def load_data(columns:[str],path="raw/distributions", *args, **kwargs):
    '''
    This file loops over the folders in the path and gets all the raw .csv. converts all the files into a single dataframe with columns specified by the columns parameter
    '''
    files = [f for f in os.listdir(path) if os.path.isfile(f)]
    for subdir in [subdir for subdir in os.listdir(path)  if not os.path.isfile(subdir)]::
        files.extend([f"{subdir}/{f}" for f in os.listdir(f"{path}/{subdir}") if os.path.isfile(f"{path}/{subdir}/{f}")])

    data = pd.concat([pd.read_csv(f"{path}/{f}", *args, **kwargs) for f in files], ignore_index=True)
    data = data[columns]
    return data
