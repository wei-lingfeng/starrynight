import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import emcee
import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool

user_path = os.path.expanduser('~')

def log_prior(theta):
    '''Log prior function: ln(p(mu)) + ln(p(sigma))
    ----------
    - Parameters:
        - theta: set of parameters to fit for: miuRA, miuDE, muvr, sigmaRA, sigmaDE, sigmavr, rho1, rho2, rho3.
    
    - Returns:
        - prior likelihood.
    '''
    if \
            all([-5 < mu  < 5 for mu  in theta[0:2]]) \
        and (mean_rv-7.5) < theta[2] < (mean_rv+7.5) \
        and all([0  < sig < 6 for sig in theta[3:6]]) \
        and all([-5 < rho < 5 for rho in theta[6:9]]):
        
        return sum([np.log(1/z) for z in theta[3:6]])
    else:
        return -np.inf
        

def log_likelihood(theta, sources):
    '''Log likelihood function. See https://arxiv.org/pdf/2105.05871.pdf Equation 4-7.
    
    Parameters
    ----------
        theta : tuple
            Set of parameters to fit for: muRA, muDE, muRV, sigmaRA, sigmaDE, sigmaRV, rho1, rho2, rho3.
        sources : pandas DataFrame with pmRA, pmRA_e, pmDE, pmDE_e, rv, e_rv
    
    Returns
    -------
        log likelihood.
    '''
    
    mu = np.array(theta[0:3])
    sigma = np.array(theta[3:6])
    rho = np.array(theta[6:9])
    
    v = np.array([sources['vRA'].value, sources['vDE'].value, sources['rv'].value]).transpose()
    epsilon = np.array([sources['e_vRA'].value, sources['e_vDE'].value, sources['e_rv'].value]).transpose()
    logL = 0
    
    for i in range(len(sources)):
        covmatrix = np.array([
            [sigma[0]**2 + epsilon[i, 0]**2,                            rho[0]*(sigma[0]*sigma[1] + epsilon[i, 0]*epsilon[i, 1]),   rho[1]*(sigma[0]*sigma[2] + epsilon[i, 0]*epsilon[i, 2])],
            [rho[0]*(sigma[0]*sigma[1] + epsilon[i, 0]*epsilon[i, 1]),  sigma[1]**2 + epsilon[i, 1]**2,                             rho[2]*(sigma[1]*sigma[2] + epsilon[i, 1]*epsilon[i, 2])],
            [rho[1]*(sigma[0]*sigma[2] + epsilon[i, 0]*epsilon[i, 2]),  rho[2]*(sigma[1]*sigma[2] + epsilon[i, 1]*epsilon[i, 2]),   sigma[2]**2 + epsilon[i, 2]**2                          ]
        ])
        
        if np.linalg.det(covmatrix) <= 0:
            return -np.inf
        else:
            logLi = -1/2*(np.log(np.linalg.det(covmatrix)) + (v[i] - mu) @ np.linalg.inv(covmatrix) @ (v[i] - mu).transpose())
        
        logL += logLi
    
    return logL


def log_posterior(theta, sources):
    return log_prior(theta) + log_likelihood(theta, sources)


def fit_vdisp(sources, save_path:str, MCMC=True, multiprocess=True, nwalkers=100, steps=500) -> dict:
    """Fit velocity dispersion for all three directions: ra, dec, radial.

    Parameters
    ----------
    sources : astropy QTable
        QTable with columns vRA, e_vRA, vDE, e_vDE, rv, e_rv.
    save_path : str
        folder to save the results.
    MCMC : bool
        run mcmc or not.
    
    Returns
    -------
    results: dict
        results[key] = [value, error]
        keys: mu_RA, mu_DE, mu_rv, sigma_RA, sigma_DE, sigma_rv, rho_RA, rho_DE, rho_rv.
        mu: velocity; sigma: intrinsic dispersion.
    """
    
    global mean_rv
    with open(f'{user_path}/ONC/starrynight/codes/analysis/vdisp_results/mean_rv.txt', 'r') as file:
        mean_rv = eval(file.read().strip('km / s'))
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    sources_new = copy.deepcopy(sources)
    sources_new = sources_new[~(np.isnan(sources_new['vRA']) | np.isnan(sources_new['vDE']) | np.isnan(sources_new['rv']))]
    
    ndim = 9
    discard = steps-200
    
    
    pos = [np.array([
        np.random.uniform(-0.5, 0.5),   # vRA
        np.random.uniform(-0.5, 0.5),   # vDE
        np.random.uniform(mean_rv - 2.5, mean_rv + 2.5),   # vr
        np.random.uniform(0, 5),        # vRA_e
        np.random.uniform(0, 5),        # vDE_e
        np.random.uniform(0, 5),        # vr_e
        np.random.uniform(-0.5, 0.5),   # rho1_e
        np.random.uniform(-0.5, 0.5),   # rho2_e
        np.random.uniform(-0.5, 0.5),   # rho3_e
    ]) for i in range(nwalkers)]
    
    
    labels = ['μ_RA', 'μ_DE', 'μ_rv', 'σ_RA', 'σ_DE', 'σ_rv', 'ρ_RA', 'ρ_DE', 'ρ_rv']
    ylabels = [r'$\mu_{RA}$', r'$\mu_{DE}$', r'$\mu_{vr}$', r'$\sigma_{RA}$', r'$\sigma_{DE}$', r'$\sigma_{vr}$', r'$\rho_{RA}$', r'$\rho_{DE}$', r'$\rho_{vr}$']
    params = ['mu_RA', 'mu_DE', 'mu_rv', 'sigma_RA', 'sigma_DE', 'sigma_rv', 'rho_RA', 'rho_DE', 'rho_rv']
    nparams = len(params)
    
    if MCMC:
        if save_path:
            backend = emcee.backends.HDFBackend(f'{save_path}/sampler.h5')
            backend.reset(nwalkers, ndim)
            
            if multiprocess:
                with Pool(32) as pool:
                    sampler=emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[sources_new], moves=emcee.moves.KDEMove(), backend=backend, pool=pool)
                    sampler.run_mcmc(pos, steps, progress=True)
            else:
                sampler=emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[sources_new], moves=emcee.moves.KDEMove(), backend=backend)
                sampler.run_mcmc(pos, steps, progress=True)
        
        else:
            if multiprocess:
                with Pool(32) as pool:
                    sampler=emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[sources_new], moves=emcee.moves.KDEMove(), pool=pool)
                    sampler.run_mcmc(pos, steps, progress=True)
            else:
                sampler=emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[sources_new], moves=emcee.moves.KDEMove())
                sampler.run_mcmc(pos, steps, progress=True)
        
        flat_samples = sampler.get_chain(discard=discard, flat=True)

        # mcmc[:, i] (2 by N) = [value, error]
        mcmc = np.empty((2, nparams))
        for i in range(nparams):
            mcmc[:, i] = np.array([np.median(flat_samples[:, i]), np.diff(np.percentile(flat_samples[:, i], [15.9, 84.1]))[0]/2])        
        
        mcmc = pd.DataFrame(mcmc, columns=params)
        
        ##################################################
        ################## Create Plots ##################
        ##################################################
        
        if save_path:
            ########## Walker Plot ##########
            fig, axes = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 18), sharex=True)
            samples = sampler.get_chain()
            
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "C0", alpha=0.2)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(ylabels[i])
            
            ax.set_xlabel("step number");
            plt.minorticks_on()
            fig.align_ylabels()
            plt.savefig(f'{save_path}/mcmc_walker.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            ########## Corner Plot ##########
            fig = corner.corner(
                flat_samples, labels=labels, truths=mcmc.loc[0].to_numpy(), quantiles=[0.16, 0.84]
            )
            plt.savefig(f'{save_path}/mcmc_corner.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            
            ########## Write Parameters ##########
            with open(f'{save_path}/mcmc_params.txt', 'w') as file:
                for label, values in zip(labels, mcmc.to_numpy().T):
                    file.write(f'{label}:\t{", ".join(str(value) for value in values)}\n')
        else:
            pass # do not generate plot if save path is none.
    
    else:
        try:
            sampler = emcee.backends.HDFBackend(f'{save_path}/sampler.h5')
        except:
            raise LookupError('Please set MCMC=True and run the sampler first before reading saved results.')
    
        flat_samples = sampler.get_chain(discard=discard, flat=True)
        
        # mcmc[:, i] (2 by N) = [value, error]
        mcmc = np.empty((2, nparams))
        for i in range(nparams):
            mcmc[:, i] = np.array([np.median(flat_samples[:, i]), np.diff(np.percentile(flat_samples[:, i], [15.9, 84.1]))[0]/2])        
        
        mcmc = pd.DataFrame(mcmc, columns=params)
    
    return mcmc