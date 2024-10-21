import numpy as np
import matplotlib.pyplot as plt
import astropy as ap
from astropy.modeling import models
import stingray.gti as stgti
import stingray as st 
import scipy as sc
from stingray import AveragedPowerspectrum, AveragedCrossspectrum, EventList
from matplotlib import cm, ticker
from astropy.io import fits


def split_lc(lc, segment_size, dt):
    """
    Splits a lightcurve into segments of length 'segment_size'.
    lc has to be a stingray lightcurve object
    """
    bins_per_seg = int(segment_size/dt)  # Then number of time bins in a given segment
    n_intervals =len(lc)//bins_per_seg # Number of intervals that a light curve is split into
    c = 0
    if n_intervals != 0:

        # Truncate the lightcurve
        temp_times = lc.time[:int(n_intervals*bins_per_seg)]
        temp_counts = lc.counts[:int(n_intervals*bins_per_seg)]
        
        # Split the light curve
        split_times = np.split(temp_times, n_intervals)

        split_counts = np.array(np.split(temp_counts, n_intervals))
        # print(split_counts.shape)
        return split_counts, split_times
    else:
        print( "Light curve cannot be split into segments for specified values")

def split_multiple_lc(lc_split, segment_size):
    """
    lc_split must be an array of stingray lightcurve objects
    segment_size is in seconds

    """
    n_stacked = 0
    c = 0
    for lc in lc_split[:]:
        split_result = split_lc(lc, segment_size, dt=lc.dt)
        
        
        if split_result is not None:
            if c == 0:
                # print('here')
                split_counts = split_result[0]
                split_times = split_result[1]
                n_stacked += 1
            elif c > 0:
                split_counts = np.append(split_counts, split_result[0], axis=0)
                split_times =  np.append(split_times, split_result[1], axis=0)
                n_stacked += 1
            pass
            c += 1
    return split_counts, split_times, n_stacked

def get_freq_indices(xf, min_freq, max_freq):
    df = xf[1] - xf[0]
    min_index = int((min_freq - xf[0]) / df)
    max_index = np.ceil((max_freq - xf[0]) / df)

    return min_index, max_index

def Load_Dat_Stingray(file_dir, energy_range, dt):
    # Load data
    eventlist = st.EventList.read(file_dir, "hea") # Load eventlist from file
    eventlist = eventlist.filter_energy_range(energy_range)
    print("Loaded Event List")

    lc_full = eventlist.to_lc(dt=1/512)
    print("Converted to LC")
    lc_gtis = lc_full.split_by_gti()

    return lc_gtis