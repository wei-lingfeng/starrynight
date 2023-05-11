import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import copy
import emcee
import corner
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
    ----------
    - Parameters:
        - theta: set of parameters to fit for: miuRA, miuDE, muvr, sigmaRA, sigmaDE, sigmavr, rho1, rho2, rho3.
        - sources: pandas DataFrame with pmRA, pmRA_e, pmDE, pmDE_e, vr, vr_e
    
    - Returns:
        - log likelihood.
    '''
    
    mu = np.array(theta[0:3])
    sigma = np.array(theta[3:6])
    rho = np.array(theta[6:9])
    
    v = np.array([sources.vRA, sources.vDE, sources.vr]).transpose()
    epsilon = np.array([sources.vRA_e, sources.vDE_e, sources.vr_e]).transpose()
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


def fit_vdisp(sources, save_path:str, MCMC=True) -> dict:
    """Fit velocity dispersion for all three directions: ra, dec, radial.

    Parameters
    ----------
    sources : pd.DataFrame
        pandas DataFrame with vRA, vRA_e, vDE, vDE_e, vr, vr_e.
    save_path : str
        folder to save the results.
    MCMC : bool
        run mcmc or not.
    
    Returns
    -------
    results: dict
        results[key] = [value, error]
        keys: mu_RA, mu_DE, mu_vr, sigma_RA, sigma_DE, sigma_vr, rho_RA, rho_DE, rho_vr.
        mu: velocity; sigma: intrinsic dispersion.
    """
    
    global mean_rv
    with open(f'{user_path}/ONC/starrynight/codes/data_processing/vdisp_results/mean_rv.txt', 'r') as file:
        mean_rv = eval(file.read())
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    sources_new = copy.deepcopy(sources)
    sources_new = sources_new.loc[~(sources_new.vRA.isna() | sources_new.vDE.isna() | sources_new.vr.isna())].reset_index(drop=True)
    
    ndim, nwalkers, step = 9, 100, 500
    discard = step-200
    
    
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
    
    
    labels = ['μ_RA', 'μ_DE', 'μ_vr', 'σ_RA', 'σ_DE', 'σ_vr', 'ρ_RA', 'ρ_DE', 'ρ_vr']
    ylabels = [r'$\mu_{RA}$', r'$\mu_{DE}$', r'$\mu_{vr}$', r'$\sigma_{RA}$', r'$\sigma_{DE}$', r'$\sigma_{vr}$', r'$\rho_{RA}$', r'$\rho_{DE}$', r'$\rho_{vr}$']
    text_labels = ['mu_RA', 'mu_DE', 'mu_vr', 'sigma_RA', 'sigma_DE', 'sigma_vr', 'rho_RA', 'rho_DE', 'rho_vr']
    
    if MCMC:
        backend = emcee.backends.HDFBackend(save_path + 'sampler.h5')
        backend.reset(nwalkers, ndim)
        
        with Pool(32) as pool:
            sampler=emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[sources_new], moves=emcee.moves.KDEMove(), backend=backend, pool=pool)
            sampler.run_mcmc(pos, step, progress=True)
        
        flat_samples = sampler.get_chain(discard=discard, flat=True)
        mcmc = np.empty((ndim, 3))
        for i in range(ndim):
            mcmc[i, :] = np.percentile(flat_samples[:, i], [16, 50, 84])
        
        # results[i, :] = [value, error]
        results = np.array([mcmc[:, 1], (mcmc[:, 2] - mcmc[:, 0])/2]).transpose()

        ##################################################
        ################## Create Plots ##################
        ##################################################
        
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
        plt.savefig(save_path + 'mcmc_walker.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        ########## Corner Plot ##########
        fig = corner.corner(
            flat_samples, labels=labels, truths=results[:, 0], quantiles=[0.16, 0.84]
        )
        plt.savefig(save_path + 'mcmc_corner.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
        ########## Write Parameters ##########
        with open(save_path + 'mcmc_params.txt', 'w') as file:
            for ylabel, values in zip(labels, results):
                file.write('{}:\t{}\n'.format(ylabel, ", ".join(str(value) for value in values)))
    
    else:
        sampler = emcee.backends.HDFBackend(save_path + 'sampler.h5')
    
        flat_samples = sampler.get_chain(discard=discard, flat=True)
        mcmc = np.empty((ndim, 3))
        for i in range(ndim):
            mcmc[i, :] = np.percentile(flat_samples[:, i], [16, 50, 84])
        
        # results[i, :] = [value, error]
        results = np.array([mcmc[:, 1], (mcmc[:, 2] - mcmc[:, 0])/2]).transpose()
    
    return {k:v for k,v in zip(text_labels, results)}