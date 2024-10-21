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

from General import *

# def split_lc(lc, segment_size, dt):
#     """
#     Splits a lightcurve into segments of length 'segment_size'.
#     lc has to be a stingray lightcurve object
#     """
#     bins_per_seg = int(segment_size/dt)  # Then number of time bins in a given segment
#     n_intervals =len(lc)//bins_per_seg # Number of intervals that a light curve is split into
#     c = 0
#     if n_intervals != 0:

#         # Truncate the lightcurve
#         temp_times = lc.time[:int(n_intervals*bins_per_seg)]
#         temp_counts = lc.counts[:int(n_intervals*bins_per_seg)]
        
#         # Split the light curve
#         split_times = np.split(temp_times, n_intervals)

#         split_counts = np.array(np.split(temp_counts, n_intervals))
#         # print(split_counts.shape)
#         return split_counts, split_times
#     else:
#         print( "Light curve cannot be split into segments for specified values")
        # return 


# def split_multiple_lc(lc_split, segment_size):
#     """
#     lc_split must be an array of stingray lightcurve objects
#     segment_size is in seconds

#     """
#     n_stacked = 0
#     c = 0
#     for lc in lc_split[:]:
#         split_result = split_lc(lc, segment_size, dt=lc.dt)
        
        
#         if split_result is not None:
#             if c == 0:
#                 # print('here')
#                 split_counts = split_result[0]
#                 split_times = split_result[1]
#                 n_stacked += 1
#             elif c > 0:
#                 split_counts = np.append(split_counts, split_result[0], axis=0)
#                 split_times =  np.append(split_times, split_result[1], axis=0)
#                 n_stacked += 1
#             pass
#             c += 1
#     return split_counts, split_times, n_stacked
    

# def make_avg_periodogram(lc_counts, lc_times, norm=True, seg_size=64, dt=1/512):
#     """
#     lc_counts is an iterable containing segmented arrays of counts
#     norm sets whetehr or not the power spectrum is normalized (fractional rms)
#     seg_size is the segment size in seconds
#     dt is the bin size of the time series for lc_counts
#     """
#     pow_list = []
#     freq_list = []
#     for counts in lc_counts:
#         # plt.plot(times, counts)

#         yf = sc.fft.fft(counts)
#         power = np.abs(yf)**2
#         xf = sc.fft.fftfreq(len(counts), dt)

#         # We need the number of photons in a segment for fractional rms normalization
#         n_photons = np.sum(counts)
#         meanctrate = n_photons / seg_size
        
#         # We are normalizing and then averaging - should this be switched?
#         norm_power = 2 / (meanctrate * n_photons) * power

#         freq_list.append(xf)
#         if norm:
#                 pow_list.append(norm_power)
#         else:
#                 pow_list.append(power)
    
#     pow_arr = np.array(pow_list)
#     avg_pow = np.sum(pow_arr, axis=0) / len(lc_counts)
    
#     return avg_pow, freq_list[0]

# def get_freq_indices(xf, min_freq, max_freq):
#     df = xf[1] - xf[0]
#     min_index = int((min_freq - xf[0]) / df)
#     max_index = np.ceil((max_freq - xf[0]) / df)

#     return min_index, max_index




# Wrappers Below

# def avg_periodogram_wrapper(data_dir, seg_size, energy_range=[3, 10], plot=True):
#     """
#     energy_range must be of the form [E_min, E_max]
#     """
#     eventlist = st.EventList.read(data_dir, "hea", additional_columns=['DET_ID']) # Load eventlist from file
#     eventlist = eventlist.filter_energy_range(energy_range)
    
#     print("Loaded Event List")

#     lc_full = eventlist.to_lc(dt=1/512)
#     print("Converted to LC")
#     lc_gtis = lc_full.split_by_gti()

#     split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=seg_size)
#     avg_pow, xf = make_avg_periodogram(split_counts, split_times)

#     if plot:
#         fig_p, ax_p = plt.subplots()

#         avg_pow = avg_pow[xf>0]
#         xf = xf[xf>0]

#         ax_p.plot(xf, avg_pow * xf, drawstyle="steps-mid", color="k", alpha=.5, ls='-.')

#         ax_p.set_yscale('log')
#         ax_p.set_xscale('log')



def bispec(ft, freq_index_min, freq_index_max, bicoherence=False):
    """
    ! freq_index_max is excluded from accessed indices

    A, B are the two terms in the denominator of the bicoherence definition
    """
    
    # Create matrix of indices
    indices = np.arange(freq_index_min, freq_index_max, 1, dtype='int')
    indices = np.tile(indices, (len(indices), 1))
    indices_sum = indices + indices.T + 1
    # print((indices == 0).any())

    # print(indices)
    # print("Indices:", indices)
    bispec_calc = ft[indices] * ft[indices.T] * np.conjugate(ft[indices_sum])
    
    if bicoherence:
        A = ft[indices] * ft[indices.T]
        B = np.conjugate(ft[indices_sum])

        return bispec_calc, A, B

    # print("Calculated bispectrum:", bispec_calc)
    
    return bispec_calc

def avg_bispec(lc_counts_list, lc_times_list, dt=1/512, min_freq=0.01, max_freq=10, bicoherence=False):
    
    print("HI")
    
    check = True # To define array on first iteration
    for counts in lc_counts_list:

        yf = sc.fft.rfft(counts)
        freq = sc.fft.rfftfreq(len(counts), dt)

        yf = yf[freq>0]
        freq = freq[freq>0]
        
        # print(yf.shape, xf.shape)
        
        # yf = yf[freq>=0]
        # freq = freq[freq>=0]        
        min_index, max_index = get_freq_indices(freq, min_freq, max_freq)

        if check:
            if bicoherence:
                avg_bispec, temp_C, temp_D = bispec(yf, min_index, max_index, bicoherence=bicoherence)
                C = np.abs(temp_C)**2
                D = np.abs(temp_D)**2
            else:
                avg_bispec = bispec(yf, min_index, max_index, bicoherence=bicoherence)
            
            check = False

        else:
            if bicoherence:
                temp_avg_bispec, temp_C, temp_D = bispec(yf, min_index, max_index, bicoherence=bicoherence)
                avg_bispec += temp_avg_bispec
                C += np.abs(temp_C)**2
                D += np.abs(temp_D)**2
            else:
                avg_bispec += bispec(yf, min_index, max_index, bicoherence=bicoherence)
    
    freq_selected = freq[int(min_index):int(max_index)]

    if bicoherence:
        b2 = np.abs(avg_bispec)**2 / (C * D)
        return b2, freq_selected
    else:
        avg_bispec = avg_bispec / len(counts)
    
    

    return avg_bispec, freq_selected
    
def avg_bispec_wrapper(data_dir, seg_size, energy_range=[3, 10], dt = 1/512, plot=True, min_freq=0.01, max_freq=10, bicoherence=False):
    
    # Load data
    # eventlist = st.EventList.read(data_dir, "hea") # Load eventlist from file
    # eventlist = eventlist.filter_energy_range(energy_range)
    # print("Loaded Event List")

    # lc_full = eventlist.to_lc(dt=1/512)
    # print("Converted to LC")
    # lc_gtis = lc_full.split_by_gti()

    lc_gtis = Load_Dat_Stingray(data_dir, energy_range, dt)

    split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=seg_size)

    if not bicoherence:
        avg_bspec, freq = avg_bispec(split_counts, split_times, min_freq=min_freq, max_freq=max_freq, bicoherence=bicoherence)
        bispec_abs = np.abs(avg_bspec)
        bispec_phase = np.arctan2(avg_bspec.imag, avg_bspec.real)
    else:
        bispec_abs, freq = avg_bispec(split_counts, split_times, min_freq=min_freq, max_freq=max_freq, bicoherence=bicoherence)
    # np.arg
    if plot:
       
        fig, ax = plt.subplots()
        cs = ax.pcolor(freq, freq, bispec_abs, norm='log', cmap='cividis')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(cs)
        ax.set_title('Magnitude')
        # plt.show()
        if not bicoherence:
            fig2, ax2 = plt.subplots()
            cs = ax2.pcolor(freq, freq, bispec_phase, cmap='cividis')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Frequency (Hz)')
            fig2.colorbar(cs)
            ax2.set_title('Phase')
        
        plt.show()
