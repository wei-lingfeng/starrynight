import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import copy
import smart
import emcee
import corner
import numpy as np
import pandas as pd
import apogee_tools as ap
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections.abc import Iterable
from scipy.interpolate import interp1d

ap.apogee_hack.tools.download.allStar(dr=17)
ap_path = '/home/l3wei/Software/apogee_data'
save_path = '/home/l3wei/ONC/Data/APOGEE/'

sources = pd.read_csv('/home/l3wei/ONC/Catalogs/sources 2d.csv')
cross_matches = (~sources.teff_nirspec.isna()) & (~sources.teff_apogee.isna())
sources_matched = sources.loc[abs(sources.loc[cross_matches].teff_nirspec - sources.loc[cross_matches].teff_apogee).sort_values(ascending=False).index].reset_index(drop=True)
apogee_ids = list(sources_matched.ID_apogee)
hc2000_ids = list(sources_matched.HC2000)
year = list(sources_matched.year.astype(int))
month = list(sources_matched.month.astype(int))
day = list(sources_matched.day.astype(int))

# apogee_ids = ['2M05351427-0524246']
apogee_ids = ['2M05351259-0523440']

params = ['teff', 'logg', 'metal', 'vsini', 'rv', 'airmass', 'pwv', 'wave_off1', 'wave_off2', 'wave_off3', 'c0_1', 'c0_2', 'c1_1', 'c1_2', 'c2_1', 'c2_2', 'noise']

priors = {
    'teff_min':2300,  'teff_max':7000,
    'logg_min':2.5,   'logg_max':5.,
    'metal_min':-2.5, 'metal_max':0.5,
    'vsini_min':0.0,  'vsini_max':100.0,
    'rv_min':-200.0,  'rv_max':200.0,
    'airmass_min':1., 'airmass_max':3.,
    'pwv_min':0.5,    'pwv_max':20,
    'wave_min':-0.5,  'wave_max':0.5,
    'c0_1_min':-10.,  'c0_1_max':10.,
    'c0_2_min':-10.,  'c0_2_max':10.,
    'c1_1_min':-10.,  'c1_1_max':10.,
    'c1_2_min':-10.,  'c1_2_max':10.,
    'c2_1_min':-10.,  'c2_1_max':10.,
    'c2_2_min':-10.,  'c2_2_max':10.,
    'noise_min':1.,   'noise_max':5.
}

limits = { 
    'teff_min':2300,    'teff_max':7200,
    'logg_min':2.5,     'logg_max':5.5,
    'metal_min':-2.5,   'metal_max':0.5,
    'vsini_min':0.0,    'vsini_max':300.0,
    'rv_min':-200.0,    'rv_max':200.0,
    'airmass_min':1.,   'airmass_max':3.,
    'pwv_min':0.5,      'pwv_max':20,
    'wave_min':-0.5,    'wave_max':0.5,
    'c0_1_min':-10000,  'c0_1_max':10000,
    'c0_2_min':-100,    'c0_2_max':100,
    'c1_1_min':-10000,  'c1_1_max':10000,
    'c1_2_min':-100,    'c1_2_max':100,
    'c2_1_min':-10000,  'c2_1_max':10000,
    'c2_2_min':-100,    'c2_2_max':100,
    'noise_min':1.,     'noise_max':10.
}

MCMC = True
Multiprocess = False
# nparams, nwalkers, steps = 17, 100, 3000
# nparams, nwalkers, steps = 17, 100, 500
nparams, nwalkers, steps = 17, 50, 100
discard = steps - 50
modelset                = 'phoenix-aces-agss-cond-2011'
instrument, order       = 'apogee', 'all'

apogee_id = apogee_ids[0]
if not os.path.exists(save_path + apogee_id):
    os.makedirs(save_path + apogee_id)

object_path = save_path + apogee_id + '/'

# ap.download(apogee_id, type='apvisit', dir=object_path, ap_path=ap_path, dr=17)

# Get the LSF
if not os.path.exists(object_path + 'lsf.npy'):
    xlsf = np.linspace(-7.,7.,43)
    lsf  = ap.apogee_hack.spec.lsf.eval(xlsf)
    with open(object_path + 'lsf.npy', 'wb') as file:
        np.save(file, xlsf)
        np.save(file, lsf)
else:
    with open(object_path + 'lsf.npy', 'rb') as file:
        xlsf = np.load(file)
        lsf = np.load(file)

n_visit = len([_ for _ in os.listdir(object_path + 'specs/') if os.path.isfile(object_path + 'specs/' + _) and _.startswith('apVisit')])
specs = []

fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
for visit in range(1, n_visit + 1):
    data_path  = object_path + 'specs/' + 'apVisit-' + apogee_id + '-{}.fits'.format(visit)
    spec = smart.Spectrum(name=apogee_id, path=data_path, instrument=instrument, applymask=True, datatype='apvisit', applytell=True)
    
    # Normalize
    spec.noise = spec.noise / np.median(spec.flux)
    spec.flux  = spec.flux  / np.median(spec.flux)
    
    specs.append(copy.deepcopy(spec))
    ax.plot(spec.wave, spec.flux, alpha=0.4, linewidth=0.2)
ax.set_xlabel('wavelength')
ax.set_ylabel('flux')
plt.show()


##################################################
################# Coadd Spectrum #################
##################################################
if n_visit > 1:
    # split spectrum
    order_border_idx = np.argsort(np.diff(specs[0].wave))[-2:][::-1]
    order_borders = [np.mean(specs[0].wave[[_, _+1]]) for _ in order_border_idx]

    # interpolate Δλ=f(λ)
    dlambda = np.diff(specs[0].wave)
    idx = dlambda < 0.2
    f_dlambda = interp1d(x=specs[0].wave[:-1][idx], y=dlambda[idx], bounds_error=False, fill_value=dlambda[-1])

    # for each order
    wave_new = np.array([])
    supersample_rate = 1
    for lower_border, upper_border in zip([-np.inf, *order_borders], [*order_borders, np.inf]):
        waves = [_.wave[(_.wave > lower_border) & (_.wave < upper_border)] for _ in specs]
        wave_min = max([min(wave) for wave in waves])
        wave_max = min([max(wave) for wave in waves])
        wave_order = [wave_min]
        while wave_order[-1] < wave_max:
            wave_order.append(wave_order[-1] + f_dlambda(wave_order[-1]) / supersample_rate)
        wave_order[-1] = wave_max
        wave_new = np.append(wave_new, np.array(wave_order))


    flux_new = np.zeros((n_visit, np.size(wave_new)))
    for i in range(n_visit):
        f_wave = interp1d(specs[i].wave, specs[i].flux)
        flux_new[i, :] = f_wave(wave_new)

    # Median Combine
    flux_med = np.median(flux_new, axis=0)

    # Update spec
    spec.wave = wave_new
    spec.flux = flux_med
    spec.noise = np.std(flux_new, axis=0)


def lnlike(theta, data, lsf, xlsf):
    """
    Log-likelihood, computed from chi-squared.

    Parameters
    ----------
    theta
    lsf
    data

    Returns
    -------
    -0.5 * chi-square + sum of the log of the noise

    """

    # Parameters MCMC
    teff, logg, metal, vsini, rv, airmass, pwv, wave_off1, wave_off2, wave_off3, c0_1, c0_2, c1_1, c1_2, c2_1, c2_2, noise = theta #A: flux offset; N: noise prefactor

    # wavelength offset is set to 0
    model = smart.makeModel(
        teff=teff, logg=logg, metal=metal, vsini=vsini, rv=rv, airmass=airmass, pwv=pwv, lsf=lsf, xlsf=xlsf,
        wave_off1=wave_off1, wave_off2=wave_off2, wave_off3=wave_off3, 
        c0_1=c0_1, c0_2=c0_2, c1_1=c1_1, c1_2=c1_2, c2_1=c2_1, c2_2=c2_2,
        instrument=instrument, order=order, modelset=modelset, data=data
    )

    chisquare = smart.chisquare(data, model)/noise**2

    return -0.5 * (chisquare + np.sum(np.log(2 * np.pi * (data.noise * noise)**2)))


def lnprior(theta, limits=limits):
    """
    Specifies a flat prior
    """
    ## Parameters for theta
    teff, logg, metal, vsini, rv, airmass, pwv, wave_off1, wave_off2, wave_off3, c0_1, c0_2, c1_1, c1_2, c2_1, c2_2, noise = theta

    if  limits['teff_min']      < teff      < limits['teff_max']\
    and limits['logg_min']      < logg      < limits['logg_max']\
    and limits['metal_min']     < metal     < limits['metal_max']\
    and limits['vsini_min']     < vsini     < limits['vsini_max']\
    and limits['rv_min']        < rv        < limits['rv_max']\
    and limits['airmass_min']   < airmass   < limits['airmass_max']\
    and limits['pwv_min']       < pwv       < limits['pwv_max']\
    and limits['wave_min']      < wave_off1 < limits['wave_max']\
    and limits['wave_min']      < wave_off2 < limits['wave_max']\
    and limits['wave_min']      < wave_off3 < limits['wave_max']\
    and limits['c0_1_min']      < c0_1      < limits['c0_1_max']\
    and limits['c0_2_min']      < c0_2      < limits['c0_2_max']\
    and limits['c1_1_min']      < c1_1      < limits['c1_1_max']\
    and limits['c1_2_min']      < c1_2      < limits['c1_2_max']\
    and limits['c2_1_min']      < c2_1      < limits['c2_1_max']\
    and limits['c2_2_min']      < c2_2      < limits['c2_2_max']\
    and limits['noise_min']     < noise     < limits['noise_max']:
        return 0.0

    return -np.inf

def lnprob(theta, data, lsf, xlsf):
        
    lnp = lnprior(theta)
        
    if not np.isfinite(lnp):
        return -np.inf
        
    return lnp + lnlike(theta, data, lsf, xlsf)

# Get the starter positions
pos = [np.array([   priors['teff_min']      + (priors['teff_max']       - priors['teff_min'])       * np.random.uniform(), 
                    priors['logg_min']      + (priors['logg_max']       - priors['logg_min'])       * np.random.uniform(), 
                    priors['metal_min']     + (priors['metal_max']      - priors['metal_min'])      * np.random.uniform(),
                    priors['vsini_min']     + (priors['vsini_max']      - priors['vsini_min'])      * np.random.uniform(),
                    priors['rv_min']        + (priors['rv_max']         - priors['rv_min'])         * np.random.uniform(),
                    priors['airmass_min']   + (priors['airmass_max']    - priors['airmass_min'])    * np.random.uniform(), 
                    priors['pwv_min']       + (priors['pwv_max']        - priors['pwv_min'])        * np.random.uniform(), 
                    priors['wave_min']      + (priors['wave_max']       - priors['wave_min'])       * np.random.uniform(), 
                    priors['wave_min']      + (priors['wave_max']       - priors['wave_min'])       * np.random.uniform(), 
                    priors['wave_min']      + (priors['wave_max']       - priors['wave_min'])       * np.random.uniform(),   
                    priors['c0_1_min']      + (priors['c0_1_max']       - priors['c0_1_min'])       * np.random.uniform(), 
                    priors['c0_2_min']      + (priors['c0_2_max']       - priors['c0_2_min'])       * np.random.uniform(), 
                    priors['c1_1_min']      + (priors['c1_1_max']       - priors['c1_1_min'])       * np.random.uniform(), 
                    priors['c1_2_min']      + (priors['c1_2_max']       - priors['c1_2_min'])       * np.random.uniform(), 
                    priors['c2_1_min']      + (priors['c2_1_max']       - priors['c2_1_min'])       * np.random.uniform(), 
                    priors['c2_2_min']      + (priors['c2_2_max']       - priors['c2_2_min'])       * np.random.uniform(),
                    priors['noise_min']     + (priors['noise_max']      - priors['noise_min'])      * np.random.uniform()
                ]) for _ in range(nwalkers)]

## multiprocessing
if MCMC:
    backend = emcee.backends.HDFBackend(object_path + 'sampler.h5')
    backend.reset(nwalkers, nparams)
    
    if Multiprocess:
        with Pool(64) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(spec, lsf, xlsf), pool=pool, moves=emcee.moves.KDEMove(), backend=backend)
            sampler.run_mcmc(pos, steps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(spec, lsf, xlsf), moves=emcee.moves.KDEMove(), backend=backend)
        sampler.run_mcmc(pos, steps, progress=True)

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    print(sampler.acceptance_fraction)

else:
    sampler = emcee.backends.HDFBackend(object_path + 'sampler.h5')


##################################################
################# Analyze Output #################
##################################################

flat_samples = sampler.get_chain(discard=discard, flat=True)

# mcmc[:, i] (3 by N) = [value, lower, upper]
mcmc = np.empty((3, nparams))
for i in range(nparams):
    mcmc[:, i] = np.percentile(flat_samples[:, i], [50, 16, 84])

mcmc = pd.DataFrame(mcmc, columns=params)


##################################################
################## Create Plots ##################
##################################################

########## Walker Plot ##########
fig, axes = plt.subplots(nrows=nparams, ncols=1, figsize=(10, 25), sharex=True)
samples = sampler.get_chain()

for i in range(nparams):
    ax = axes[i]
    ax.plot(samples[:, :, i], "C0", alpha=0.2)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(params[i])

ax.set_xlabel("Step Number");
plt.minorticks_on()
fig.align_ylabels()
plt.savefig(object_path + 'MCMC_Walker.png', dpi=300, bbox_inches='tight')
plt.close()

########## Corner Plot ##########
fig = corner.corner(
    flat_samples, labels=params, truths=mcmc.loc[0].to_numpy(), quantiles=[0.16, 0.84]
)
plt.savefig(object_path + 'MCMC_Corner.png', dpi=300, bbox_inches='tight')
plt.savefig(object_path + 'MCMC_Corner.pdf', dpi=300, bbox_inches='tight')
plt.close()


##################################################
################# Writing Result #################
##################################################
# 'teff', 'logg', 'metal', 'vsini', 'rv', 'airmass', 'pwv', 'waveoffset_1', 'waveoffset_2', 'waveoffset_3' 'c0_1', 'c0_2', 'c1_1', 'c1_2', 'c2_1', 'c2_2', 'N'
result = {
    'APOGEE_ID':    apogee_id,
    'teff':         mcmc.teff,
    'logg':         mcmc.logg,
    'metal':        mcmc.metal,
    'vsini':        mcmc.vsini,
    'rv':           mcmc.rv, 
    'airmass':      mcmc.airmass, 
    'pwv':          mcmc.pwv, 
    'wave_off1':    mcmc.wave_off1,
    'wave_off2':    mcmc.wave_off2,
    'wave_off3':    mcmc.wave_off3,
    'c0_1':         mcmc.c0_1,
    'c0_2':         mcmc.c0_2,
    'c1_1':         mcmc.c1_1,
    'c1_2':         mcmc.c1_2,
    'c2_1':         mcmc.c2_1,
    'c2_2':         mcmc.c2_2,
    'noise':        mcmc.noise,
    'snr':          np.median(spec.flux/spec.noise)
}

########## Write Parameters ##########
with open(object_path + 'MCMC_Params.txt', 'w') as file:
    for key, value in result.items():
        if isinstance(value, Iterable) and (not isinstance(value, str)):
            file.write('{}: \t{}\n'.format(key, ", ".join(str(_) for _ in value)))
        else:
            file.write('{}: \t{}\n'.format(key, value))