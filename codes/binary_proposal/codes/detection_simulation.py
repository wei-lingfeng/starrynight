from __future__ import division
import os
from tqdm import tqdm
import velbin
import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing.pool import Pool
from contextlib import closing


def detectable_distribution(fbin, n_targets=258, n_repeats=1e4, times=[0, 2], precision=0.5):
    '''
    Calculate distribution of the number of detectable binaries.
    
    Parameters:
    - fbin: binary fraction
    - n_targets: number of observed targets.
    - n_repeats: repetition
    - times: Observation epoch (year)
    - precision: RV measurement precision in km/s.
    
    Returns:
    - mean_1s, std_1s: 1-sigma mean of number of detectable binaries.
    - mean_3s, std_3s: 3-sigma mean of number of detectable binaries.
    '''
    
    n_binaries = int(1e6)  # total number of binaries in the sample
    n_stars = int(5e4)     # number of stars to select from sample
    n_repeats = int(n_repeats)
    mass_lo, mass_hi = 0.1, 1.5 # solar masses
    
    detected = np.empty(n_repeats)  # fraction of detected binaries in each iteration.

    for i in tqdm(range(n_repeats)):
        
        # print('Running...{:.2f}%'.format(i/n_repeats*100))
        
        random.seed()
        masses = np.random.uniform(mass_lo, mass_hi, n_binaries)
        all_binaries = velbin.solar(nbinaries=n_binaries)
        all_binaries.draw_mass_ratio('flat')

        index = (all_binaries.semi_major(masses) < 58.35)
        selected_binaries = all_binaries[index][:n_stars]
        masses = masses[index][:n_stars]

        is_bin = np.random.rand(n_stars) < fbin
        rvs = np.array([selected_binaries.velocity(mass=masses, time=time)[0,:] for time in times])
        delta_rv = rvs[1] - rvs[0]
        delta_rv[~is_bin] = 0


        # randomly sample n_target stars.
        observed_index = random.sample(range(n_stars), n_targets)
        # nbins = sum(is_bin[observed_index])
        detected[i] = sum(abs(delta_rv[observed_index]) > 3 * precision)
        
    
    # print('Finished!')
    
    return detected



def my_boxplot(positions, datas, fill=False, width=None, ax=None, **kwargs):
    '''Box Plot of detectable binary fraction vs imposed binary fraction.
    
    Capped lines: min, max values.
    Boxes: 16- and 84th percentile.
    Center line: Median
    
    Parameters:
    - positions: iterable of x-axis position.
    - datas: iterable of datas corresponding to each position.
    '''
    
    if len(positions) != len(datas):
        raise ValueError('Length of positions ({:d}) and datas ({:d}) are not consistent.'.format(len(positions), len(datas)))
    
    if not width:
        width = np.median(-np.diff(np.sort(positions)))/2
    
    if not ax:
        ax = plt.gca()
    
    
    for position, data in zip(positions, datas):
        # Calculate percentiles [16, 50, 84]%
        percentiles = np.percentile(data, [16, 50, 84])
        # Draw capped lines
        ax.hlines([min(data), max(data)], position - width/3, position + width/3, zorder=0, **kwargs)
        ax.vlines(position, min(data), max(data), zorder=0, **kwargs)
        
        # Draw rectangles
        # white background
        rect_background = patches.Rectangle(
            (position - width/2, percentiles[0]), 
            width, 
            percentiles[2] - percentiles[0],
            facecolor='white',
            edgecolor='none',
        )
        
        if fill:
            rect = patches.Rectangle(
                (position - width/2, percentiles[0]), 
                width, 
                percentiles[2] - percentiles[0],
                alpha=0.5,
                **kwargs
            )
        
        else:
            rect = patches.Rectangle(
                (position - width/2, percentiles[0]), 
                width, 
                percentiles[2] - percentiles[0],
                edgecolor=kwargs['color'],
                facecolor='none',
                linewidth=1.5
            )
        
        ax.add_patch(rect_background)
        ax.add_patch(rect)
        # Draw center line
        ax.hlines(percentiles[1], position - width/2, position + width/2, colors='C1', linewidth=2.5)
    



if __name__ == '__main__':
    
    new_run = False
    fbins = np.arange(0.1, 1.1, 0.1)
    
    if new_run:
        with closing(Pool()) as pool:
            results = np.array(pool.map(detectable_distribution, fbins))
            pool.terminate()
        
        with open('detection_simulation.npy', 'wb') as file:
            np.save(file, results)
    
    else:
        with open('detection_simulation.npy', 'rb') as file:
            results = np.load(file)
        
    # # 1-sigma plot
    # fig, ax = plt.subplots(figsize=(8, 4.5))
    # my_boxplot(fbins, result_1s, fill=False, ax=ax, color='C0')
    # ax.set_xticks(np.arange(0.1, 1.1, 0.1))
    # ax.set_xticklabels(np.arange(0.1, 1.1, 0.1))
    # ax.set_xlabel('Binary Fraction', fontsize=12)
    # ax.set_ylabel('Detectable Binary Sources', fontsize=12)
    # plt.savefig('boxplot_1sigma.png', dpi=300)
    # plt.show()
    
    # 3-sigma plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    my_boxplot(fbins*100, results, fill=False, ax=ax, color='C0')
    ax.set_xticks(np.arange(10, 110, 10))
    ax.set_xticklabels(np.arange(10, 110, 10))
    ax.set_title('Binary Detection Simulation for 258 Sources')
    ax.set_xlabel('Binary Fraction (%)', fontsize=12)
    ax.set_ylabel('Detectable Binary Pairs', fontsize=12)
    plt.savefig('Figures/boxplot_3sigma.pdf', dpi=300)
    plt.show()