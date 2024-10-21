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

import scipy.stats as scst

from General import *

def make_avg_periodogram(lc_counts, lc_times, norm=True, seg_size=64, dt=1/512):
    """
    lc_counts is an iterable containing segmented arrays of counts
    norm sets whetehr or not the power spectrum is normalized (fractional rms)
    seg_size is the segment size in seconds
    dt is the bin size of the time series for lc_counts
    """
    pow_list = []
    freq_list = []
    for counts in lc_counts:
        # plt.plot(times, counts)

        yf = sc.fft.fft(counts)
        power = np.abs(yf)**2
        xf = sc.fft.fftfreq(len(counts), dt)

        # We need the number of photons in a segment for fractional rms normalization
        n_photons = np.sum(counts)
        meanctrate = n_photons / seg_size
        
        # We are normalizing and then averaging - should this be switched?
        norm_power = 2 / (meanctrate * n_photons) * power

        freq_list.append(xf)
        if norm:
                pow_list.append(norm_power)
        else:
                pow_list.append(power)
    
    pow_arr = np.array(pow_list)
    avg_pow = np.sum(pow_arr, axis=0) / len(lc_counts)
    
    return avg_pow, freq_list[0]

def rebin(x, y, f, n_stacked=0):
    """
    x must be linearly spaced
    """
    dx = x[1] - x[0]

    # create new set of bins:
    minx = 0
    maxx = x[-1]
    new_bins = [minx, minx+dx]
    while new_bins[-1] <= maxx:
        dx *= (1+f)
        # print(new_bins)
        new_bins.append(new_bins[-1] + dx)

    new_bins = np.asanyarray(new_bins)
    # print(new_bins)

    rebinned_x, edges_x, indices_x = scst.binned_statistic(x, x, 'mean', new_bins)
    # print(bins_x)
    rebinned_y_real, edges_real, indices_real = scst.binned_statistic(x, y.real, 'mean', new_bins)

    n_binned = [np.count_nonzero(indices_real[indices_real==i]) for i in range(len(rebinned_y_real))]
    n_binned = np.asanyarray(n_binned)

    # print(indices_real)
    # print(rebinned_x)
# 
    return rebinned_x, rebinned_y_real, n_binned + n_stacked

def avg_periodogram_wrapper(data_dir, seg_size, energy_range=[3, 10], plot=True, rebin_f=0):
    """
    energy_range must be of the form [E_min, E_max]
    """
    eventlist = st.EventList.read(data_dir, "hea") # Load eventlist from file
    eventlist = eventlist.filter_energy_range(energy_range)
    
    print("Loaded Event List")

    lc_full = eventlist.to_lc(dt=1/512)
    print("Converted to LC")
    lc_gtis = lc_full.split_by_gti()

    split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=seg_size)
    avg_pow, freq = make_avg_periodogram(split_counts, split_times)

    avg_pow = avg_pow[freq>0]
    freq = freq[freq>0]

    rebinned_freq, rebinned_pow, total_stacked = rebin(freq, avg_pow, f=rebin_f, n_stacked=10)

    if plot:
        fig_p, ax_p = plt.subplots()
        ax_p.plot(freq, avg_pow * freq, drawstyle="steps-mid", color="k", alpha=.5, ls='-.')
        ax_p.plot(rebinned_freq, rebinned_pow * rebinned_freq)

        ax_p.set_yscale('log')
        ax_p.set_xscale('log')

        ax_p.set_xlabel('Frequency (Hz)')
        ax_p.set_ylabel('Power (Units)')
        plt.show()
    
    return rebinned_pow, rebinned_freq, total_stacked

