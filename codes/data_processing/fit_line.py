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
    

    if outlier_rejection:
        theta, b, pb, xb, yb, vx, vy = params
        if \
            -np.pi/2        <=  theta   <= np.pi/2      \
        and -np.inf         <   b       < np.inf        \
        and 0               <   pb      < 1             \
        and prior['xb'][0]  <   xb      < prior['xb'][1]\
        and prior['yb'][0]  <   yb      < prior['yb'][1]\
        and prior['vx'][0]  <   vx      < prior['vx'][1]\
        and prior['vy'][0]  <   vy      < prior['vy'][1]:
            return 0.0
        else:
            return -np.inf
    
    else:
        theta, b = params
        if \
            -np.pi/2    <=  theta   <= np.pi/2           \
        and -np.inf     <   b       < np.inf:
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
    
           
    # xerr and yerr with outlier rejection
    if outlier_rejection:
        theta, b, pb, xb, yb, vx, vy = params
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
        theta, b = params
        v = np.array([-np.sin(theta), np.cos(theta)]).transpose()
        Z = np.array([x, y]).transpose()
        S = np.array([np.diag([xerr[i]**2, yerr[i]**2]) for i in range(len(x))])
        logL = 0
        for i in range(len(x)):
            logL += -(np.dot(v.T, Z[i]) - b*np.cos(theta))**2 / (2*(v.T@S[i]@v))
            
    return logL


def log_probability(params, x, y, xerr, yerr, prior, outlier_rejection):
    if np.isinf(log_prior(params, prior, outlier_rejection)):
        return -np.inf
    else:
        return log_likelihood(params, x, y, xerr, yerr, outlier_rejection)



def fit_line(x, y, xerr, yerr, prior, save_path=None, outlier_rejection=False, nwalkers=100, step=500, discard=400, MCMC=True, Multiprocess=False, **kwargs):
    """Fit the slope k and intercept b of the form y = kx + b to data with xerr and yerr.
    The sampler turns the slope into inclination theta first, i.e., k = tan(theta).
    Theta is allowed to vary from -π/2 to π/2, and b is allowed to vary from -∞ to +∞.
    The prior parameter can control the initial distribution of k (or theta) and b.

    Parameters
    ----------
    x : 1-D array
        x data
    y : 1-D array
        y data
    xerr : 1-D array
        x error
    yerr : 1-D array
        y error
    prior : dict
        Prior distribution range of k (or theta) (optional) and b.
        e.g.: {'k': np.array([-5, 5]), 'b': np.array([-5, 5])}.
        Note that this does not set a limit on the prior likelihood function.
    save_path : str, optional
        Save path, by default None
    outlier_rejection : bool, optional
        Automatically reject outliers or not, by default False
    nwalkers : int, optional
        Number of walkers, by default 100
    step : int, optional
        Number of steps, by default 500
    discard : int, optional
        The begining number of steps to discard, by default 400
    MCMC : bool, optional
        Run MCMC or not, by default True
    Multiprocess : bool, optional
        Turn on multiprocessing or not, by default False

    Returns
    -------
    results : pd.DataFrame
        mcmc results of the form [value, error]. 
        By default, results have keys ['theta', 'b', 'k']. 
        If outlier=True, the keys are ['theta', 'b', 'pb', 'xb', 'yb', 'vx', 'vy', 'k']
        e.g., results['k'] = array([k, k_err])
    """
    
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
    
    if 'b' not in prior.keys():
        raise ValueError("prior must specify the prior distribution range of the intercept 'b'.")
    
    if 'k' in prior.keys():
        prior['theta'] = np.arctan(prior['k'])
    elif ('theta' not in prior.keys()) and ('k' not in prior.keys()):
        prior['theta'] = np.array([-np.pi/2, np.pi/2])
    
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        pass
    
    # discard = step - 100
    
    if outlier_rejection:
        params = ['theta', 'b', 'pb', 'xb', 'yb', 'vx', 'vy']
        prior['xb'] = [xmin, xmax]
        prior['yb'] = [ymin, ymax]
        prior['vx'] = [0, xmax - xmin]
        prior['vy'] = [0, ymax - ymin]
        pos = [np.array([
            np.random.uniform(prior['theta'][0], prior['theta'][1]),
            np.random.uniform(prior['b'][0], prior['b'][1]),
            np.random.uniform(0, 1),
            np.random.uniform(prior['xb'][0], prior['xb'][1]),
            np.random.uniform(prior['yb'][0], prior['yb'][1]),
            np.random.uniform(prior['vx'][0], prior['vx'][1]),
            np.random.uniform(prior['vy'][0], prior['vy'][1])
        ]) for _ in range(nwalkers)]
    else:
        params = ['theta', 'b']
        pos = [np.array([
            np.random.uniform(prior['theta'][0], prior['theta'][1]),
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
    
    # mcmc[:, i] (2 by N) = [value, error]
    mcmc = np.empty((2, ndim))
    for i in range(ndim):
        percentiles = np.percentile(flat_samples[:, i], [50, 16, 84])
        mcmc[:, i] = np.array([percentiles[0], (percentiles[2] - percentiles[1])/2])
    
    mcmc = pd.DataFrame(mcmc, columns=params)
    
        
    ##################################################
    ################## Create Plots ##################
    ##################################################
    
    ########## Walker Plot ##########
    if outlier_rejection:
        fig, axes = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 10), sharex=True)
    else:
        fig, axes = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 5), sharex=True)
    
    samples = sampler.get_chain()
    
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "C0", alpha=0.2)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(params[i])

    ax.set_xlabel("step number");
    plt.minorticks_on()
    fig.align_ylabels()
    if save_path:
        plt.savefig(save_path + '/MCMC Walker.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    ########## Corner Plot ##########
    fig = corner.corner(
        flat_samples, labels=params, truths=mcmc.loc[0].to_numpy(), quantiles=[0.16, 0.84]
    )
    if save_path:
        plt.savefig(save_path + '/MCMC Corner.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    ########## Convert to slope ##########
    mcmc['k'] = np.array([np.tan(mcmc.theta[0]), mcmc.theta[1] / np.cos(mcmc.theta[0])])

    ########## Write Parameters ##########
    if save_path:
        with open(save_path + '/MCMC Params.txt', 'w') as file:
            for param, values in mcmc.items():
                file.write('{}:\t{}\n'.format(param, ", ".join(str(value) for value in values)))
    
    return mcmc





if __name__=='__main__':
    save_path = '/home/l3wei/ONC/Codes/Data Processing/line/function/'
    # data = pd.read_table('/home/l3wei/ONC/Codes/Data Processing/line/line.txt', sep=' ', names=['ID', 'x', 'y', 'yerr', 'xerr', 'rho_xy'], usecols=[1,2,3,4,5])
    data = pd.read_csv('/home/l3wei/ONC/Codes/Data Processing/line/mass vs vrel.csv')
    
    result = fit_line(data.mass, data.vrel, data.mass_e, data.vrel_e, save_path=save_path, outlier_rejection=False, MCMC=True, Multiprocess=True)