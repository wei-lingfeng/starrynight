# -*- coding: utf-8 -*-
# Generate binary simulation data using python 2 for later plotting.
from __future__ import division
import os
import velbin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm

user_path = os.path.expanduser('~')

def simulate_binaries(sources, fbin, n_sims, show_figure=False):
    n_sims = int(n_sims)

    sources = sources.loc[sources.theta_orionis.isna()].reset_index(drop=True)

    N = len(sources)
    print('N={}'.format(N))
    rv = sources.rv
    rv_err = sources.rv_e
    mass_MIST = sources.loc[~sources.mass_MIST.isna(), 'mass_MIST']

    # sort the data
    rv_err_sorted = np.sort(rv_err)
    mass_MIST_sorted = np.sort(mass_MIST)

    # calculate the porportional values of samples
    p_rv_err = 1. * np.arange(len(rv)) / (len(rv) - 1)
    p_mass_MIST = 1. * np.arange(len(mass_MIST)) / (len(mass_MIST) - 1)

    # interpolate inverse cdf
    # cdf_rv_err(rv_err_sorted) = p_rv_err
    # cdf_rv_err(mass_MIST_sorted) = p_mass_MIST
    inv_cdf_rv_err = interp1d(p_rv_err, rv_err_sorted)
    inv_cdf_mass_MIST = interp1d(p_mass_MIST, mass_MIST_sorted)

    # Parameters
    dates = (0., )
    sigma = 2

    with open(user_path + '/ONC/starrynight/codes/data_processing/vdisp_results/all/mcmc_params.txt', 'r') as file:
        raw = file.readlines()
    raw = [line for line in raw if line.startswith('σ_rv:')][0]
    vdisp_rv, vdisp_rv_e = eval(raw.strip('σ_rv:\t\n'))

    limit_low, limit_high = 1, 4.5

    # Simulation
    valid_sims = 0
    v_dispersions = np.empty(n_sims)

    print('Start generating data for fbin={}...'.format(fbin))
    while valid_sims < n_sims:
        all_binaries = velbin.solar(nbinaries=N)
        all_binaries.draw_mass_ratio('flat')
        all_binaries.draw_eccentricities()

        # masses = np.random.uniform(mass_low, mass_high, N)
        masses = inv_cdf_mass_MIST(np.random.uniform(low=0, high=1., size=N))

        velocities = all_binaries.velocity(masses)
        
        vdisp = np.random.uniform(low=limit_low, high=limit_high, size=1)
        sigvel = inv_cdf_rv_err(np.random.uniform(low=0, high=1., size=N))

        # fake dataset
        v_systematic = np.random.randn(N) * vdisp
        v_bin_offset = np.array([all_binaries[:N].velocity(masses, time)[0, :] for time in dates])
        v_bin_offset[:, np.random.rand(N) > fbin] = 0.
        v_meas_offset = np.random.randn(v_bin_offset.size).reshape(v_bin_offset.shape) * np.atleast_1d(sigvel)
        
        mock_dataset = np.squeeze(v_systematic[np.newaxis, :] + v_bin_offset + v_meas_offset)
        v_intrinsic = v_systematic
        v_binary = np.squeeze(v_bin_offset)
        v_errors = np.squeeze(v_meas_offset)
        v_syst_errors = np.squeeze(v_systematic[np.newaxis, :] + v_meas_offset)

        mean_mock, std_mock = norm.fit(mock_dataset[abs(mock_dataset) <= 7])
        # mean_mock, std_mock = norm.fit(mock_dataset)
        
        if abs(std_mock - vdisp_rv) > vdisp_rv_e * sigma:
            continue
        
        v_dispersions[valid_sims] = vdisp
        valid_sims += 1
    
    # Plot
    if show_figure & (fbin==0.5):
        rv_offset = rv - np.mean(rv)
        mean_obs, std_obs = norm.fit(rv_offset[abs(rv_offset) <= 7])
        x = np.linspace(-10, 10, 1000)
        y_mock = norm.pdf(x, mean_mock, std_mock)
        y_intrinsic = norm.pdf(x, 0, vdisp)
        y_obs = norm.pdf(x, mean_obs, std_obs)


        bins = np.linspace(-10, 10, 20)
        linewidth = 1.5
        alpha = 0.7
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(mock_dataset, color='C0', bins=bins, histtype='step', density=True, alpha=alpha, linewidth=linewidth)
        ax.hist(rv - np.mean(rv), color='C7', label='Observed', bins=bins, histtype='step', density=True, alpha=alpha, linewidth=linewidth)
        ax.hist(v_errors, color='C2', label='Errors', bins=bins, histtype='step', density=True, alpha=alpha, linestyle='-.', linewidth=linewidth)
        ax.hist(v_binary, color='C6', label=r'Binaries ({0:.0%})'.format(fbin), bins=bins, histtype='step', density=True, alpha=alpha, linestyle='--', linewidth=linewidth)
        ax.plot(x, y_intrinsic, color='C3', label='Intrinsic', linestyle=':', linewidth=linewidth*1.2)
        ax.plot(x, y_mock, color='C0', label='Intrinsic +' + '\n' + 'Binaries +'  + '\n' + 'Errors', linewidth=1)
        ax.plot(x, y_obs, color='C7', label='Observed', linewidth=1)
        ax.legend(fontsize=12)
        ax.set_ylim((0, 0.2))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel(r'$\Delta v_r$ $\left(\mathrm{km}\cdot\mathrm{s}^{-1}\right)$', fontsize=15)
        ax.set_ylabel('Normalized Distribution', fontsize=15)
        plt.savefig(user_path + '/ONC/figures/Binary Simulation Histogram.pdf')
        plt.show()
    
    else:
        with open('{}/ONC/starrynight/codes/binary_simulation/v_disp fbin={:.2f}.npy'.format(user_path, fbin), 'wb') as file:
            np.save(file, v_dispersions)
        print('Simulation of fbin {:.0%} is now finished!'.format(fbin))
    return v_dispersions


n_sims = 1e5
fbins = np.linspace(0, 1, 5, endpoint=True)
show_figure = False

# n_sims = 1
# fbins=[0.5]
# show_figure = True

sources = pd.read_csv(user_path + '/ONC/starrynight/catalogs/sources 2d.csv')
v_dispersions = [simulate_binaries(sources=sources, fbin=fbin, n_sims=n_sims, show_figure=show_figure) for fbin in fbins]