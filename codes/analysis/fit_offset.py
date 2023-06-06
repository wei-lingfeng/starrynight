import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def log_prior(params, prior, outlier_rejection):
    
    # yerr ONLY with outlier rejection
    # theta, b, pb, yb, vb = params
    # if \
    #     np.arctan(1.7)  < theta < np.arctan(2.8) \
    # and 10              < b     < 60 \
    # and 0               < pb    < 1 \
    # and 200             < yb    < 600 \
    # and 0               < vb    < 100:
    #     return 0.0
    # else:
    #     return -np.inf
    
    
    # xerr and yerr with outlier rejection
    if outlier_rejection:
        b, pb, xb, yb, vx, vy = params
        if \
            prior['b'][0]       < b     < prior['b'][1]     \
        and 0                   < pb    < 1                 \
        and prior['xb'][0]      < xb    < prior['xb'][1]    \
        and prior['yb'][0]      < yb    < prior['yb'][1]    \
        and prior['vx'][0]      < vx    < prior['vx'][1]    \
        and prior['vy'][0]      < vy    < prior['vy'][1]:
            return 0.0
        else:
            return -np.inf
    
    else:
        b = params[0]
        if prior['b'][0]       < b     < prior['b'][1]:
            return 0.0
        else:
            return -np.inf


def log_likelihood(params, x, y, xerr, yerr, outlier_rejection):
    
    # yerr ONLY with outlier rejection
    # theta, b, pb, yb, vb = params
    # m = np.tan(theta)
    
    # logL = sum(np.log(
    #     (1-pb)/np.sqrt(2*np.pi*yerr**2) * np.exp(-(y - m*x - b)**2/(2*yerr**2)) +\
    #     pb/np.sqrt(2*np.pi*(vb + yerr**2)) * np.exp(-(y - yb)**2/(2*(vb**2 + yerr**2)))
    # ))
    
    theta = np.pi/4
    # xerr and yerr with outlier rejection
    if outlier_rejection:
        b, pb, xb, yb, vx, vy = params
        v = np.array([-np.sin(theta), np.cos(theta)])
        Z = np.array([x, y]).transpose()
        S = np.array([np.diag([xerr[i]**2, yerr[i]**2]) for i in range(len(x))])
        b_xy = np.array([xb, yb])
        sigma_xy = np.diag([vx**2, vy**2])
        logL = 0
        for i in range(len(x)):
            logL += np.log(
                (1-pb) * np.exp(-(np.dot(v, Z[i]) - b*np.cos(theta))**2 / (2*(v@S[i]@v))) +\
                pb / (2*np.pi*np.sqrt(np.linalg.det(sigma_xy))) * np.exp(-1/2 * (Z[i] - b_xy) @ np.linalg.inv(sigma_xy) @ (Z[i] - b_xy))
            )
            
    else:
        b = params[0]
        v = np.array([-np.sin(theta), np.cos(theta)])
        Z = np.array([x, y]).transpose()
        S = np.array([np.diag([xerr[i]**2, yerr[i]**2]) for i in range(len(x))])
        logL = 0
        for i in range(len(x)):
            logL += -(np.dot(v, Z[i]) - b*np.cos(theta))**2 / (2*(v@S[i]@v))
            
    return logL


def log_probability(params, x, y, xerr, yerr, prior, outlier_rejection):
    if np.isinf(log_prior(params, prior, outlier_rejection)):
        return -np.inf
    else:
        return log_likelihood(params, x, y, xerr, yerr, outlier_rejection)



def fit_offset(x, y, xerr, yerr, prior, save_path=None, outlier_rejection=False, nwalkers=100, step=500, discard=400, MCMC=True, Multiprocess=False):
    '''Fit a line y=x+b to data with xerr and yerr.
    --------------------
    - Parameters:
        - x, y, xerr, yerr: 1-D array-like data.
        - prior: array-like prior of b: [b_min, b_max].
    - Optional Parameters:
        - save_path: path to save the fitting results. If None: don't save.
        - outlier_rejection: fit with outlier rejection or not.
        - nwalkers: number of walkers of emcee ensemble sampler.
        - step: steps of emcee ensemble sampler.
        - discard: number of steps at the begining to discard.
        - MCMC: whether run mcmc or read from backend file (requires a run with MCMC=True first). True or False.
        - Multiprocess: whether run with multiprocessing or not.
    - Returns:
        - fit_result: [b, b_low, b_high].
    '''
    
    C0 = '#1f77b4'
    C1 = '#ff7f0e'
    C3 = '#d62728'
    C4 = '#9467bd'
    C6 = '#e377c2'
    C7 = '#7f7f7f'
    C9 = '#17becf'
    
    x = np.array(x)
    y = np.array(y)
    xerr = np.array(xerr)
    yerr = np.array(yerr)
    
    constraint = ~(np.isnan(x) | np.isnan(y) | np.isnan(xerr) | np.isnan(yerr))
    
    x = x[constraint]
    y = y[constraint]
    xerr = xerr[constraint]
    yerr = yerr[constraint]
    
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    
    prior = {'b': prior}
    prior['xb'] = [xmin, xmax]
    prior['yb'] = [ymin, ymax]
    prior['vx'] = [0, xmax - xmin]
    prior['vy'] = [0, ymax - ymin]    
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        pass
    
    # discard = step - 100
    
    if outlier_rejection:
        pos = [np.array([
            np.random.uniform(prior['b'][0], prior['b'][1]),
            np.random.uniform(0, 1),
            np.random.uniform(prior['xb'][0], prior['xb'][1]),
            np.random.uniform(prior['yb'][0], prior['yb'][1]),
            np.random.uniform(prior['vx'][0], prior['vx'][1]),
            np.random.uniform(prior['vy'][0], prior['vy'][1])
        ]) for _ in range(nwalkers)]
    else:
        pos = [np.array([
            np.random.uniform(prior['b'][0], prior['b'][1])
        ]) for _ in range(nwalkers)]
    
    ndim = len(pos[0])
    
    
    if MCMC:
        if save_path:
            backend = emcee.backends.HDFBackend(save_path + '/sampler.h5')
            backend.reset(nwalkers, ndim)
        else:
            backend=None
        
        if Multiprocess:
            with Pool(32) as pool:
                sampler=emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[x, y, xerr, yerr, prior, outlier_rejection], moves=emcee.moves.KDEMove(), backend=backend, pool=pool)
                sampler.run_mcmc(pos, step, progress=True)
        else:
            sampler=emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[x, y, xerr, yerr, prior, outlier_rejection], moves=emcee.moves.KDEMove(), backend=backend)
            sampler.run_mcmc(pos, step, progress=True)
    
    else:
        sampler = emcee.backends.HDFBackend(save_path + '/sampler.h5')
    
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    mcmc = np.empty((ndim, 3))
    for i in range(ndim):
        mcmc[i, :] = np.percentile(flat_samples[:, i], [16, 50, 84])
    
    # mcmc[i, :] = [value, lower, upper]
    mcmc = np.array([mcmc[:, 1], mcmc[:, 0], mcmc[:, 2]]).transpose()
    
    ##################################################
    ################## Create Plots ##################
    ##################################################
    x0, x1 = min(x), max(x)
    inds = np.random.randint(len(flat_samples), size=100)
    fig, ax = plt.subplots(figsize=(6, 4))
    for ind in inds:
        offset = flat_samples[ind, 0]
        ax.plot([x0, x1], [x0+offset, x1+offset], color='C1', alpha=0.1)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.')
    ax.plot([min(y), x1], [min(y), x1], color='C3', label='Equal Line', linestyle='dashed')
    ax.plot([x0, x1], [x0+mcmc[0, 0], x1+mcmc[0, 0]], color='k', label='Offset Line')
    ax.legend()
    ax.set_xlabel('Kim pmRA (mas/yr)')
    ax.set_ylabel('Gaia pmRA (mas/yr)')
    plt.savefig(save_path + '/offset fit.png')
    plt.show()
    
    if outlier_rejection:
        # xerr and yerr without outlier rejection
        ylabels = ['b', 'pb', 'xb', 'yb', 'vx', 'vy']
    else:
        # xerr and yerr without outlier rejection
        ylabels = ['b']
    
    ########## Walker Plot ##########
    if outlier_rejection:
        fig, axes = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 8), sharex=True)
    else:
        fig, ax = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 3), sharex=True)
    
    samples = sampler.get_chain()
    
    for i in range(ndim):
        if outlier_rejection:
            ax = axes[i]
        ax.plot(samples[:, :, i], "C0", alpha=0.2)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(ylabels[i])

    ax.set_xlabel("step number");
    plt.minorticks_on()
    fig.align_ylabels()
    if save_path:
        plt.savefig(save_path + '/MCMC Walker.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    ########## Corner Plot ##########
    fig = corner.corner(
        flat_samples, labels=ylabels, truths=mcmc[:, 0], quantiles=[0.16, 0.84]
    )
    if save_path:
        plt.savefig(save_path + '/MCMC Corner.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    ########## Write Parameters ##########
    if save_path:
        with open(save_path + '/MCMC Params.txt', 'w') as file:
            for ylabel, values in zip(ylabels, mcmc):
                file.write('{}:\t{}\n'.format(ylabel, ", ".join(str(value) for value in values)))
    
    return mcmc[0]



if __name__=='__main__':
    save_path = '/home/l3wei/ONC/Codes/Data Processing/'
    sources = pd.read_csv('/home/l3wei/ONC/Catalogs/synthetic catalog - epoch combined.csv')

    rv_constraint = ~(
        (abs(sources.rv_corrected_nirspec) > 20) | 
        (abs(sources.rv_corrected_apogee) > 20)
    )
    sources = sources[rv_constraint].reset_index(drop=True)
    offset = fit_offset(sources.pmRA_kim, sources.pmRA_gaia, xerr=sources.pmRA_e_kim, yerr=sources.pmRA_e_gaia, prior=[0, 2], save_path=save_path + 'offset fit/', outlier_rejection=False)