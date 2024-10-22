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
        # We need to calculate additional values for the bicoherence
        A = ft[indices] * ft[indices.T]
        B = np.conjugate(ft[indices_sum])

        return bispec_calc, A, B
    
    else:
        return bispec_calc

def avg_bispec(lc_counts_list, lc_times_list, dt=1/512, min_freq=0.01, max_freq=10, bicoherence=False):

    
    check = True # To define array on first iteration
    for counts in lc_counts_list:

        yf = sc.fft.rfft(counts)
        freq = sc.fft.rfftfreq(len(counts), dt)

        yf = yf[freq>0]
        freq = freq[freq>0]

        if check:
            
            # This needs to be done only once since frequencies will be the same for all FTs
            min_index, max_index = get_freq_indices(freq, min_freq, max_freq)
            
            if bicoherence:
                # Define values if first iter
                avg_bispec, temp_C, temp_D = bispec(yf, min_index, max_index, bicoherence=bicoherence)
                C = np.abs(temp_C)**2
                D = np.abs(temp_D)**2
            else:
                # Add values if not first iter
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
    
def avg_bispec_wrapper(data_dir, seg_size, energy_range=[3, 10], dt = 1/512, plot=True, min_freq=0.01, max_freq=10, bicoherence=False, savefig=None):
    
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
        ax.set_title('Absolute')

        if savefig != None:
            fig.savefig(f'{savefig}_abs.png')
        # plt.show()
        if not bicoherence:
            fig2, ax2 = plt.subplots()
            cs = ax2.pcolor(freq, freq, bispec_phase, cmap='cividis')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Frequency (Hz)')
            fig2.colorbar(cs)
            ax2.set_title('Phase')
            if savefig != None:
                fig2.savefig(f'{savefig}_phase.png')
        
        # plt.show()
    return freq, bispec_abs


def gen_bispectra_bootstrapping(lc_counts_list, dt=1/512, min_freq=0.01, max_freq=10, bicoherence=False):
    """
    In order to use bootstrapping, we want to be able to sample from the set of bispectra that we are creating from the segmented lightcurve.
    Hence, this needs to be stored as an array/list and returned.
    """

    
    check = True # To check for first iteration

    bispec_list = [] # Stores all the computed bispectra

    for counts in lc_counts_list:
        
        # rfft only gives real component of FT
        yf = sc.fft.rfft(counts)
        freq = sc.fft.rfftfreq(len(counts), dt)

        # Get rid of the 0 frequency
        yf = yf[freq>0]
        freq = freq[freq>0]

        

        if check:
            
            # This needs to be done only once since frequencies will be the same for all FTs
            min_index, max_index = get_freq_indices(freq, min_freq, max_freq)
            check = False

        else:
            if bicoherence:
                # A way to use bootstrapping with the bicoherence is not currently defined
                pass
            
            else:
                bispec_list.append(bispec(yf, min_index, max_index, bicoherence=bicoherence))
    
    freq_selected = freq[int(min_index):int(max_index)]

    if bicoherence:
        print("You should not be here!")
        pass

    # else:
    #     avg_bispec = avg_bispec / len(counts)
    
    bispec_arr = np.asanyarray(bispec_list)

    return bispec_arr, freq_selected

def get_phase_from_list(bispec_arr, QPO_bin_num):
    """
    Calculate an average bispectrum from a list of bispectra
    """

    avg_bispec = np.sum(bispec_arr, axis=0) / len(bispec_arr)
    average_phase = np.angle(avg_bispec)[QPO_bin_num, QPO_bin_num]
    
    return average_phase


def sample_bispec(bispec_arr, n_samples=None):
    """
    Samples an array of bispectra to create a new population of bispectra
    """
    if n_samples is None:
        n_samples = len(bispec_arr)
    sampled_indices = np.random.choice(len(bispec_arr), n_samples)
    sample = bispec_arr[sampled_indices]



    return sample


def wrapper_phase(data_dir, seg_size, QPO_bin, lc_gtis = None, split_counts=None, n_bootstrapping_iters=2000, energy_range=[3, 10], dt = 1/512, plot=True, savefig=None):
    
    # Add the option to directly input gtis to save time on loading data
    if lc_gtis is None:
        lc_gtis = Load_Dat_Stingray(data_dir, energy_range, dt)
    
    if split_counts is None:
        split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=seg_size)

    # Generate bispectra to sample from
    bispec_arr, freq = gen_bispectra_bootstrapping(split_counts)

    phase_list = []
    for i in range(n_bootstrapping_iters):

        new_sample = sample_bispec(bispec_arr)
        phase_list.append(get_phase_from_list(bispec_arr=new_sample, QPO_bin_num=QPO_bin))
    
    phase_mean = np.mean(phase_list)
    phase_std = np.std(phase_list)

    return phase_mean, phase_std