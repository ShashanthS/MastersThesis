from BiSpectra import *
from PowerSpec import *
# from BiSpec_NoNoise import *
# Quickest file to run: 


# file_dir = 'C:/Users/shash/UvA\Thesis/Project/MastersThesis/data/raw/SWIFT/ni6203980102_0mpu7_cl.evt'
evt_file_name = 'ni1050360109_0mpu7_cl'
evt_file_name = 'ni1130360110_0mpu7_cl'

file_dir = f'C:/Users/shash/UvA/Thesis/Project/MastersThesis/data/raw/MAXI/{evt_file_name}.evt'

seg_size_periodogram = 32
e_range=[3,10]

save_name = f'C:/Users/shash/UvA/Thesis/Project/MastersThesis/data/processed/22102024_PhaseCalcs/{evt_file_name}_21102024'

full_model = lmfit.models.LorentzianModel(prefix='fund_') + lmfit.models.LorentzianModel(prefix='harm_') + lmfit.models.LorentzianModel(prefix='Bbn1_') + lmfit.models.LorentzianModel(prefix='Bbn2_') + lmfit.models.ConstantModel(prefix='poisson_')

params=lmfit.Parameters()

params.add('fund_amplitude', value=0.01, min=0)
params.add('fund_center', value=4.4, min=0)
params.add('fund_sigma', value=1, min=0)

params.add('harm_center', expr='2.0*fund_center')
params.add('harm_amplitude', value=0.003, min=0)
params.add('harm_sigma', value=1, min=0)

params.add('Bbn1_amplitude', value=0.01, min=0)
params.add('Bbn1_center', value=.3, min=0)
params.add('Bbn1_sigma', value=0.5, min=0)

params.add('Bbn2_amplitude', value=0.01, min=0)
params.add('Bbn2_center', value=0, min=0)
params.add('Bbn2_sigma', value=10, min=0)

params.add('poisson_c', value=0.01, min=0)

# Create Power Spectrum
freq, avg_pow, stacked  = avg_periodogram_wrapper(file_dir, seg_size_periodogram, energy_range=e_range, plot=False, rebin_f=0.02)

# Fit the powerspec
fit_res = fit_powerspec(freq[1:], avg_pow[1:], plot_fit=True, save_name=save_name, model=full_model, params=params)

# Get fundamental frequency from result of fit and define bin size using it
fund_cent = fit_res.params['fund_center'].value
factor = 8
bin_size = 1/fund_cent * factor

lc_gtis = Load_Dat_Stingray(file_dir, e_range, dt=1/256)
split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=bin_size)

# Create Bispectrum
freq_bspec, abs_bspec = avg_bispec_wrapper(file_dir, bin_size, lc_gtis=lc_gtis, split_counts=split_counts, min_freq=0.1, max_freq=20, bicoherence=True, energy_range=e_range, savefig=save_name, plot=True)

# Confirm that the fundamental frequency is at the accessed bin
print('Check whether we are accessing the correct bin:', freq_bspec[factor-1], fund_cent)

phase_mean, phase_std = wrapper_phase(file_dir, bin_size, factor-1, lc_gtis = lc_gtis, split_counts=split_counts)
print(f'The calculated phase for {evt_file_name} is {phase_mean / np.pi:.3f}π ± {phase_std:.3f}π')

plt.show()























# plt.axvline((freq_bspec[factor-1] + freq_bspec[factor])/2, color='green', linestyle='dashed')
# plt.axvline((freq_bspec[factor-1] + freq_bspec[factor-2])/2, color='green', linestyle='dashed')

# plt.axvline((freq_bspec[2*factor-1] + freq_bspec[2*factor])/2, color='red', linestyle='dashed')
# plt.axvline((freq_bspec[2*factor-1] + freq_bspec[2*factor-2])/2, color='red', linestyle='dashed')

# ax_p.axvline(freq_bspec[factor-1], color='green', linestyle='dotted')
# ax_p.axvline(freq_bspec[2*factor-1] , color='red', linestyle='dotted')
# ax_p.axvline(fund_cent, color='black')
# print((freq_bspec[factor-1] + freq_bspec[factor])/2)

# print('Ratio of fund to harmonic:', freq_bspec[2*factor-1] / fund_cent)