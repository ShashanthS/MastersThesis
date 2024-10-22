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

from BiSpectra import get_freq_indices


def split_lc_nn(lc_A, lc_B, segment_size, dt):
    """
    Splits a lightcurve into segments of length 'segment_size'.
    lc_A, lc_B have to be a stingray lightcurve object
    dt is the time resolution of the lightcurve in seconds
    """   
    bins_per_seg = int(segment_size/dt)  # Then number of time bins in a given segment
    n_intervals =np.min((len(lc_A)//bins_per_seg, len(lc_B)//bins_per_seg))  # Number of intervals that a light curve is split into

    if n_intervals != 0:

        # Truncate the lightcurve
        temp_times_A = lc_A.time[:int(n_intervals*bins_per_seg)]
        temp_counts_A = lc_A.counts[:int(n_intervals*bins_per_seg)]

        temp_times_B = lc_B.time[:int(n_intervals*bins_per_seg)]
        temp_counts_B= lc_B.counts[:int(n_intervals*bins_per_seg)]
        
        # Split the light curve components
        split_times_A =  np.array(np.split(temp_times_A, n_intervals))
        split_counts_A = np.array(np.split(temp_counts_A, n_intervals))

        split_times_B =  np.array(np.split(temp_times_B, n_intervals))
        split_counts_B = np.array(np.split(temp_counts_B, n_intervals))
        
        if split_counts_A.shape == split_counts_B.shape == split_times_A.shape == split_times_B.shape:
            return split_counts_A, split_counts_B, split_times_A, split_times_B
        else:
            print("The split arrays have different shapes for some reason!")
    
    else:
        print( "Light curve cannot be split into segments for specified values")
        # return 


def avg_bispec_nn(lc_counts_A, lc_counts_B, dt=1/512, min_freq=0.01, max_freq=10):
    
    
    check = True # To define array on first iteration
    for counts_A, counts_B in zip(lc_counts_A, lc_counts_B):
        
        ft_A = sc.fft.fft(counts_A)
        ft_B = sc.fft.fft(counts_B)
        # yf = sc.fft.fft(counts)

        freq = sc.fft.fftfreq(len(counts_A), dt) # Common for both sets of data
        
        # We only care abt positive frequencies
        ft_A = ft_A[freq>0]
        ft_B = ft_B[freq>0]
        freq = freq[freq>0] 
     
        min_index, max_index = get_freq_indices(freq, min_freq, max_freq)

        if check:
            avg_bispec = bispec_nn(ft_A, ft_B, min_index, max_index, bicoherence=False)
            check = False

        else:
            avg_bispec += bispec_nn(ft_A, ft_B, min_index, max_index, bicoherence=False)
            # print(avg_bispec)
    
    avg_bispec = avg_bispec / len(counts_A)
    freq_selected = freq[int(min_index):int(max_index)]
    
    return avg_bispec, freq_selected


def load_split_data(dir = 'C:/Users/shash/UvA/Thesis/Project/MastersThesis/data/raw/MAXI/ni1050360104_0mpu7_cl.evt'):
    f = fits.open(dir)

    FPM_arr = np.asanyarray([[i for i in range(8)] , [i for i in range(10, 18)] , [i for i in range(20, 28)] , [i for i in range(30, 38)] , [i for i in range(40, 48)] , [i for i in range(50, 58)] , [i for i in range(60, 68)]])
    
    FPMA_IDs = np.array([FPM_arr[0], FPM_arr[3], FPM_arr[5]]).flatten()
    FPMB_IDs = np.array([FPM_arr[1], FPM_arr[2], FPM_arr[4], FPM_arr[6]]).flatten()

    selected_times_A = f[1].data['TIME'][np.isin(f[1].data['DET_ID'], FPMA_IDs)]
    selected_e_A = f[1].data['PI'][np.isin(f[1].data['DET_ID'], FPMA_IDs)] * 0.01

    ev_A = st.EventList(selected_times_A)
    ev_A.energy = selected_e_A
    ev_A.gti = np.array(f[2].data.tolist())

    selected_times_B = f[1].data['TIME'][np.isin(f[1].data['DET_ID'], FPMB_IDs)]
    selected_e_B = f[1].data['PI'][np.isin(f[1].data['DET_ID'], FPMB_IDs)] * 0.01

    ev_B = st.EventList(selected_times_B)
    ev_B.energy = selected_e_B
    ev_B.gti = np.array(f[2].data.tolist())
    return ev_A, ev_B


def bispec_nn(ft_A, ft_B, freq_index_min, freq_index_max, bicoherence=False):
    indices = np.arange(freq_index_min, freq_index_max, 1, dtype='int')
    indices = np.tile(indices, (len(indices), 1))
    indices_sum = indices + indices.T

    bispec_calc = ft_A[indices] * ft_A[indices.T] * np.conjugate(ft_B[indices_sum])

    if bicoherence:
        # To Do - add calculating and returning bicoherence here
        pass
    
    else:

        return bispec_calc