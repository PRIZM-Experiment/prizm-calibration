'''This python file regroups all functions related to computing calibration factors using on-sky 
measurements. 
We are planning a future separate file for calibration factors computed from internal calibrators.'''

# Import modules
import numpy as np
import os
import statistics as stat
import math
import scipy as sci
from matplotlib import pyplot as plt
import PrizmCalibration as cl # this module is missing dependencies
from gsm_data import GSMData, get_desired_frequencies


# -------------------- WORK IN PROGRESS ---------------------------- #
# What functions do we need?

# Step 1: Convolve the GSM model with the antenna beam. This gives the expected temperature T_GSM.

# Step 2: Compute/find P_sky from the data. P_sky is the antenna power coming from just the sky.

# Step 3: Compute the transmission efficiency η (eta) from the data.

# STEP REMOVED (Step 4: For a full day of real data and GSM simulated data, take a full day average and remove 
# the time independent component.)

# Step 5: Do a fit for K_dGSM at each frequency to get K_dGSM(nu).

# Step 6: Multiply the data P_sky by K_dGSM to get the calibrated data.

def compute_k_sky():
    '''Computes the sky calibration factor.'''
    return



'''Template'''
def function2():
    return

# -------------------- WORK IN PROGRESS ---------------------------- #