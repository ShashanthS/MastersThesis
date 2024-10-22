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
import lmfit
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
    
    return rebinned_freq, rebinned_pow, total_stacked


## Fitting Functions

def obj_fcn(params, data_x, data_y, model, n_stacked):
    """
    Log likelihood function to be used for fitting.
    
    """
    # m = PDS_reb.m # Number of stacked periodograms
    # data_x = xdat 
    # data_y = ydat
    model_y = model.eval(params=params, x=data_x)
    
    S = 2 * np.sum(n_stacked * (data_y/model_y + np.log(model_y) + (1/n_stacked - 1) * np.log(data_x) + 100*n_stacked))
    
    # print(S)

    return S


def fit_powerspec(data_x, data_y, plot_fit=True, save_name=None):
    # Define a Model:
    full_model = lmfit.models.LorentzianModel(prefix='fund_') + lmfit.models.LorentzianModel(prefix='harm_') + lmfit.models.LorentzianModel(prefix='Bbn1_') + lmfit.models.LorentzianModel(prefix='Bbn2_') + lmfit.models.ConstantModel(prefix='poisson_')

    # Add Params
    params=lmfit.Parameters()

    params.add('fund_amplitude', value=0.01, min=0)
    params.add('fund_center', value=2.13, min=0)
    params.add('fund_sigma', value=0.1, min=0)

    params.add('harm_center', expr='2.0*fund_center')
    params.add('harm_amplitude', value=0.003, min=0)
    params.add('harm_sigma', value=0.3, min=0)

    params.add('Bbn1_amplitude', value=0.01, min=0)
    params.add('Bbn1_center', value=.3, min=0)
    params.add('Bbn1_sigma', value=0.5, min=0)

    params.add('Bbn2_amplitude', value=0.01, min=0)
    params.add('Bbn2_center', value=0, min=0)
    params.add('Bbn2_sigma', value=10, min=0)

    params.add('poisson_c', value=0.1, min=0)

    result = lmfit.minimize(obj_fcn, params, method='nelder', nan_policy='raise', calc_covar=True, args=(data_x, data_y, full_model, 1))

    # print(result)

    if plot_fit:
        fig_powspec, ax_powspec = plt.subplots()
        for mod in full_model.components:
            model_pow = mod.eval(result.params, x=data_x)
            ax_powspec.plot(data_x, model_pow* data_x, linestyle='dashed', label=mod.name)
        
        model_pow = full_model.eval(result.params, x=data_x)   
        ax_powspec.plot(data_x, model_pow * data_x, linestyle='dashed', label='Full Model')
        ax_powspec.plot(data_x, data_y * data_x, drawstyle="steps-mid", color="k", alpha=.5, ls='-.', label='Data')
        
        ax_powspec.set_xscale('log')
        ax_powspec.set_yscale('log')
        fig_powspec.legend()

        if save_name is not None:
             fig_powspec.savefig(f'{save_name}_powerspecfit.png')
    
    return result

