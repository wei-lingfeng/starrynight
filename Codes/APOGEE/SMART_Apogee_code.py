import sys, os, os.path, time, gc
import numpy as np
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
import matplotlib.gridspec as gridspec
import emcee
from multiprocessing import Pool
#import nirspec_fmp as nsp
import corner
import copy
import json
import ast
import warnings
warnings.filterwarnings("ignore")
import apogee_tools as ap
import smart
#import apogee as ap

topbase                = os.getenv("HOME")
save_BASE              = topbase+'/ONC/Data/APOGEE/2M05351427-0524246/'
data_BASE              = topbase+'/ONC/Data/APOGEE/2M05351427-0524246/specs/'

os.environ["OMP_NUM_THREADS"] = "1" # Just set this once at the top

log_path_failed = save_BASE + '/failed_sources.txt'

list_of_APOGEE_IDs = ['2M05351427-0524246'] # PUT YOUR APOGEE IDs HERE


## priors NEED TO UPDATE THIS FOR BINARIES
priors                 =  { 'teff_min':2300,  'teff_max':7000,
                            'logg_min':2.5,   'logg_max':5.,
                            'metal_min':-2.5, 'metal_max':0.5,
                            'vsini_min':0.0,  'vsini_max':100.0,
                            'rv_min':-200.0,  'rv_max':200.0,
                            'alpha_min':0.5,  'alpha_max':0.9,
                            'am_min':1.,      'am_max':3.,
                            'pwv_min':0.5,    'pwv_max':20,
                            'wave_min':-0.5,  'wave_max':0.5,
                            'A_min':-10.,     'A_max':10.,
                            'c0_1_min':-10.,  'c0_1_max':10.,
                            'c0_2_min':-10.,  'c0_2_max':10.,
                            'c1_1_min':-10.,  'c1_1_max':10.,
                            'c1_2_min':-10.,  'c1_2_max':10.,
                            'c2_1_min':-10.,  'c2_1_max':10.,
                            'c2_2_min':-10.,  'c2_2_max':10.,
                            'B_min':-10.,     'B_max':10.,
                            'N_min':1.,       'N_max':5.           }


# set up the input data list
MCMC                    = True
Multiprocess            = False
Fraction                = 0.3
ndim, nwalkers, step    = 17, 100, 3000
burn                    = step-100 # mcmc sampler chain burn
ndim2, nwalkers2, step2 = 17, 100, 600
burn2                   = step2-100 # mcmc sampler chain burn
moves                   = 2.0
applymask               = True
pixel_start, pixel_end  = 0, -1
plot_show               = False
custom_mask             = []
outlier_rejection       = 3
lw                      = 0.5
time1, time2            = 100, 10000000

instrument, order       = 'apogee', 'all'
#modelset                = 'btsettl08' #'btsettl08' 'phoenixaces'
modelset                = 'phoenix-aces-agss-cond-2011'

teff_min_limit         = 2300
teff_max_limit         = 7000

# Get the LSF
if not os.path.exists(save_BASE + 'lsf.npy'):
    xlsf = np.linspace(-7.,7.,43)
    lsf  = ap.apogee_hack.spec.lsf.eval(xlsf)
    with open(save_BASE + 'lsf.npy', 'wb') as file:
        np.save(file, xlsf)
        np.save(file, lsf)
else:
    with open(save_BASE + 'lsf.npy', 'rb') as file:
        xlsf = np.load(file)
        lsf = np.load(file)

for name in list_of_APOGEE_IDs:
    
    for visit in range(1,20): # If there are multiple visits

        print(name, visit)

        path  = data_BASE + 'apVisit-' + name + '-{}.fits'.format(visit)

        # ## check if the data is downloaded
        # print('1')
        # try:
        #     AP_PATH = topbase+'/repos/apogee_data'
        #     ap.download(name, type='apvisit', dir=data_BASE, ap_path=AP_PATH, dr=16)
        # except:
        #     file_log_failed = open(log_path_failed,"w+")
        #     file_log_failed.write("name %s, visit %s, 1"%(name, visit))
        #     file_log_failed.close()
        #     continue
        
        print('2')
        data       = smart.Spectrum(name=name, path=path, instrument='apogee', applymask=True, datatype='apvisit', applytell=True)
        try:
            data       = smart.Spectrum(name=name, path=path, instrument='apogee', applymask=True, datatype='apvisit', applytell=True)
        except:        
            file_log_failed = open(log_path_failed,"w+")
            file_log_failed.write("name %s, visit %s, 2"%(name, visit))
            file_log_failed.close()
            continue

        pixels     = np.arange(len(data.flux))
        data1      = copy.deepcopy(data)

        source                  = str(data.header['OBJID'])

        save_to_path            = save_BASE + '/apvisit/'
        save_to_pathcheck       = save_BASE + '/apvisit/'

        # Check if we did this fit already
        if os.path.exists(save_to_pathcheck + '/sampler_chain.npy'): 
            continue

        print('3')
        print(save_to_path)
        #print(data.header)
        #print(data.header10)
        # print(data.LSF)
        #sys.exit()
        # barycentric corrction
        barycorr = smart.barycorr(data.header, instrument='apogee').value
        print("barycorr: {} km/s".format(barycorr))

        if save_to_path is not None:
            if not os.path.exists(save_to_path):
                os.makedirs(save_to_path)
        else:
            save_to_path = '.'




        if MCMC:

            log_path = save_to_path + '/mcmc_parameters.txt'
            file_log = open(log_path,"w+")
            file_log.write("data_path {} \n".format(data.path))
            file_log.write("data_name {} \n".format(source))
            file_log.write("custom_mask {} \n".format(custom_mask))
            file_log.write("priors {} \n".format(priors))
            file_log.write("ndim {} \n".format(ndim))
            file_log.write("nwalkers {} \n".format(nwalkers))
            file_log.write("step {} \n".format(step))
            file_log.write("burn {} \n".format(burn))
            file_log.write("pixel_start {} \n".format(pixel_start))
            file_log.write("pixel_end {} \n".format(pixel_end))
            file_log.write("barycorr {} \n".format(barycorr))
            #file_log.write("med_snr {} \n".format(med_snr))
            file_log.close()


            ## read the input custom mask and priors
            lines          = open(save_to_path+'/mcmc_parameters.txt').read().splitlines()
            custom_mask    = json.loads(lines[2].split('custom_mask')[1])
            priors         = ast.literal_eval(lines[3].split('priors ')[1])


            # no logg 5.5 for teff lower than 900
            if priors['teff_min'] <= 1300: logg_max = 5.0
            else: logg_max = 5.5

            # limit of the flux nuisance parameter: 5 percent of the median flux
            A_const       = 0.05 * abs(np.median(data.flux))

            if modelset == 'btsettl08':
                limits         = { 
                                    'teff_min':max(priors['teff_min']-500,500), 'teff_max':min(priors['teff_max']+500,3500),
                                    'logg_min':3.5,                             'logg_max':logg_max,
                                    'vsini_min':0.0,                            'vsini_max':20.0,
                                    'rv_min':-200.0,                            'rv_max':200.0,
                                    'alpha_min':0.1,                            'alpha_max':4.0,
                                    'A_min':-A_const,                           'A_max':A_const,
                                    'N_min':1.,                                 'N_max':10.0                
                                }

            elif modelset == 'phoenix-aces-agss-cond-2011':
                limits         = { 
                                    'teff_min':max(priors['teff_min']-200,2300), 'teff_max':min(priors['teff_max']+200,7000),
                                    'logg_min':2.5,                              'logg_max':logg_max,
                                    'metal_min':-2.5,                            'metal_max':0.5,
                                    'vsini_min':0.0,                             'vsini_max':300.0,
                                    'rv_min':-200.0,                             'rv_max':200.0,
                                    'alpha_min':0.1,                             'alpha_max':5.1,
                                    'am_min':1.,                                 'am_max':3.,
                                    'pwv_min':0.5,                               'pwv_max':20,
                                    'wave_min':-0.5,                             'wave_max':0.5,
                                    'c0_1_min':-10000,                           'c0_1_max':10000,
                                    'c0_2_min':-100,                             'c0_2_max':100,
                                    'c1_1_min':-10000,                           'c1_1_max':10000,
                                    'c1_2_min':-100,                             'c1_2_max':100,
                                    'c2_1_min':-10000,                           'c2_1_max':10000,
                                    'c2_2_min':-100,                             'c2_2_max':100,
                                    'A_min':-A_const,                            'A_max':A_const,
                                    'N_min':1.,                                  'N_max':10.                
                                }



            #########################################################################################
            ## for multiprocessing
            #########################################################################################
            def makeModel(teff,logg,z,vsini,rv,am,pwv,wave_off1,wave_off2,wave_off3,c0_1,c0_2,c1_1,c1_2,c2_1,c2_2,**kwargs):
                """
                Return a forward model.

                Parameters
                ----------
                teff   : effective temperature
                
                data   : an input science data used for continuum correction

                Optional Parameters
                -------------------
                

                Returns
                -------
                model: a synthesized model
                """

                # read in the parameters
                order      = kwargs.get('order', 'all')
                modelset   = kwargs.get('modelset', 'btsettl08')
                instrument = kwargs.get('instrument', 'nirspec')
                lsf        = kwargs.get('lsf')   # instrumental LSF
                xlsf       = kwargs.get('xlsf')   # instrumental LSF
                tell       = kwargs.get('tell', True) # apply telluric
                data       = kwargs.get('data', None) # for continuum correction and resampling

                #print()
                #print(teff, logg)
                

                model    = smart.Model(teff=teff, logg=logg, z=z, modelset=modelset, instrument=instrument, order=order)
                #print('1', model.flux)
                # Dirty fix here
                model.wave = model.wave[np.where(model.flux != 0)]
                model.flux = model.flux[np.where(model.flux != 0)]
                #print('2', model.flux)

    
                #print('1', model.flux)

                # apply vmicro
                vmicro = 2.478 - 0.325*logg
                model.flux = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vmicro, rotate=False, gaussian=True)
                #print('2', model.flux)

                # apply vsini
                model.flux = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True, gaussian=False)
                #print('3', model.flux)      
                
                # apply rv (including the barycentric correction)
                model.wave = rvShift(model.wave, rv=rv)

                # flux veiling
                #model.flux += veiling
                
                # apply telluric
                if tell is True:
                    model = smart.applyTelluric(model=model, airmass=am, pwv=pwv)
                #print('4', model.flux)
                
                # instrumental LSF
                model.flux = ap.apogee_hack.spec.lsf.convolve(model.wave, model.flux, lsf=lsf, xlsf=xlsf).flatten()
                model.wave = ap.apogee_hack.spec.lsf.apStarWavegrid()
                # Remove the NANs
                model.wave = model.wave[~np.isnan(model.flux)]
                model.flux = model.flux[~np.isnan(model.flux)]
                #print('5', model.flux)


                # add a fringe pattern to the model
                #model.flux *= (1+amp*np.sin(freq*(model.wave-phase)))

                # wavelength offset
                #model.wave += wave_offset


                # integral resampling
                if data is not None:
                    
                    # contunuum correction
                    deg         = 5

                    ## because of the APOGEE bands, continuum is corrected from three pieces of the spectra
                    data0       = copy.deepcopy(data)
                    model0      = copy.deepcopy(model)
            
                    # wavelength offset
                    model0.wave += wave_off1

                    range0      = np.where((data0.wave >= data.oriWave0[0][-1]) & (data0.wave <= data.oriWave0[0][0]))
                    data0.wave  = data0.wave[range0]
                    data0.flux  = data0.flux[range0]
                    if data0.wave[0] > data0.wave[-1]:
                        data0.wave = data0.wave[::-1]
                        data0.flux = data0.flux[::-1]
                    model0.flux = np.array(smart.integralResample(xh=model0.wave, yh=model0.flux, xl=data0.wave))
                    model0.wave = data0.wave
                    model0      = smart.continuum(data=data0, mdl=model0, deg=deg)
                    # flux corrections
                    model0.flux = (model0.flux + c0_1) * np.e**(-c0_2)
                    
                    data1       = copy.deepcopy(data)
                    model1      = copy.deepcopy(model)

                    # wavelength offset
                    model1.wave += wave_off2

                    range1      = np.where((data1.wave >= data.oriWave0[1][-1]) & (data1.wave <= data.oriWave0[1][0]))
                    data1.wave  = data1.wave[range1]
                    data1.flux  = data1.flux[range1]
                    if data1.wave[0] > data1.wave[-1]:
                        data1.wave = data1.wave[::-1]
                        data1.flux = data1.flux[::-1]
                    model1.flux = np.array(smart.integralResample(xh=model1.wave, yh=model1.flux, xl=data1.wave))
                    model1.wave = data1.wave
                    model1      = smart.continuum(data=data1, mdl=model1, deg=deg)
                    # flux corrections
                    model1.flux = (model1.flux + c1_1) * np.e**(-c1_2)
                    
                    data2       = copy.deepcopy(data)
                    model2      = copy.deepcopy(model)

                    # wavelength offset
                    model2.wave += wave_off3

                    range2      = np.where((data2.wave >= data.oriWave0[2][-1]) & (data2.wave <= data.oriWave0[2][0]))
                    data2.wave  = data2.wave[range2]
                    data2.flux  = data2.flux[range2]
                    if data2.wave[0] > data2.wave[-1]:
                        data2.wave = data2.wave[::-1]
                        data2.flux = data2.flux[::-1]
                    model2.flux = np.array(smart.integralResample(xh=model2.wave, yh=model2.flux, xl=data2.wave))
                    model2.wave = data2.wave
                    model2      = smart.continuum(data=data2, mdl=model2, deg=deg)
                    # flux corrections
                    model2.flux = (model2.flux + c2_1) * np.e**(-c2_2)

                    ## scale the flux to be the same as the data
                    #model0.flux *= (np.std(data0.flux)/np.std(model0.flux))
                    #model0.flux -= np.median(model0.flux) - np.median(data0.flux)

                    #model1.flux *= (np.std(data1.flux)/np.std(model1.flux))
                    #model1.flux -= np.median(model1.flux) - np.median(data1.flux)

                    #model2.flux *= (np.std(data2.flux)/np.std(model2.flux))
                    #model2.flux -= np.median(model2.flux) - np.median(data2.flux)

                    model.flux  = np.array( list(model2.flux) + list(model1.flux) + list(model0.flux) )
                    model.wave  = np.array( list(model2.wave) + list(model1.wave) + list(model0.wave) )

                # flux corrections
                #model.flux        = (model.flux + c0) * np.e**(-c1) + c2

                # flux offset
                #model.flux += flux_offset
                #model.flux **= (1 + flux_exponent_offset)

                return model

            def rvShift(wavelength, rv):
                """
                Perform the radial velocity correction.

                Parameters
                ----------
                wavelength  :   numpy array 
                                model wavelength (in Angstroms)

                rv          :   float
                                radial velocity shift (in km/s)

                Returns
                -------
                wavelength  :   numpy array 
                                shifted model wavelength (in Angstroms)
                """
                return wavelength * ( 1 + rv / 299792.458)


            #########################################################################################
            ## for multiprocessing
            #########################################################################################

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

                ## Parameters MCMC
                teff, logg, mh, vsini, rv, am, pwv, wave_off1, wave_off2, wave_off3, c0_1, c0_2, c1_1, c1_2, c2_1, c2_2, N = theta #A: flux offset; N: noise prefactor

                ## wavelength offset is set to 0
                model = makeModel(teff, logg, mh, vsini, rv, am, pwv, wave_off1, wave_off2, wave_off3, c0_1, c0_2, c1_1, c1_2, c2_1, c2_2,
                    lsf=lsf, xlsf=xlsf, data=data, modelset=modelset, instrument=instrument, order=order)

                chisquare = smart.chisquare(data, model)/N**2

                return -0.5 * (chisquare + np.sum(np.log(2*np.pi*(data.noise*N)**2)))

            def lnprior(theta, limits=limits):
                """
                Specifies a flat prior
                """
                ## Parameters for theta
                teff, logg, mh, vsini, rv, am, pwv, wave_off1, wave_off2, wave_off3, c0_1, c0_2, c1_1, c1_2, c2_1, c2_2, N = theta

                if  limits['teff_min']  < teff         < limits['teff_max'] \
                and limits['logg_min']  < logg         < limits['logg_max'] \
                and limits['metal_min'] < mh           < limits['metal_max'] \
                and limits['vsini_min'] < vsini        < limits['vsini_max']\
                and limits['rv_min']    < rv           < limits['rv_max']   \
                and limits['am_min']    < am           < limits['am_max']\
                and limits['pwv_min']   < pwv          < limits['pwv_max']\
                and limits['wave_min']  < wave_off1    < limits['wave_max']\
                and limits['wave_min']  < wave_off2    < limits['wave_max']\
                and limits['wave_min']  < wave_off3    < limits['wave_max']\
                and limits['c0_1_min']  < c0_1         < limits['c0_1_max']\
                and limits['c0_2_min']  < c0_2         < limits['c0_2_max']\
                and limits['c1_1_min']  < c1_1         < limits['c1_1_max']\
                and limits['c1_2_min']  < c1_2         < limits['c1_2_max']\
                and limits['c2_1_min']  < c2_1         < limits['c2_1_max']\
                and limits['c2_2_min']  < c2_2         < limits['c2_2_max']\
                and limits['N_min']     < N            < limits['N_max']:
                    return 0.0

                return -np.inf

            def lnprob(theta, data, lsf, xlsf):
                    
                lnp = lnprior(theta)
                    
                if not np.isfinite(lnp):
                    return -np.inf
                    
                return lnp + lnlike(theta, data, lsf, xlsf)

            # Get the starter positions
            pos = [np.array([   priors['teff_min']  + (priors['teff_max']   - priors['teff_min'] ) * np.random.uniform(), 
                                priors['logg_min']  + (priors['logg_max']   - priors['logg_min'] ) * np.random.uniform(), 
                                priors['metal_min'] + (priors['metal_max']  - priors['metal_min'] )* np.random.uniform(),
                                priors['vsini_min'] + (priors['vsini_max']  - priors['vsini_min']) * np.random.uniform(),
                                priors['rv_min']    + (priors['rv_max']     - priors['rv_min']   ) * np.random.uniform(),
                                priors['am_min']    + (priors['am_max']     - priors['am_min'])    * np.random.uniform(), 
                                priors['pwv_min']   + (priors['pwv_max']    - priors['pwv_min'])   * np.random.uniform(), 
                                priors['wave_min']  + (priors['wave_max']   - priors['wave_min'])  * np.random.uniform(), 
                                priors['wave_min']  + (priors['wave_max']   - priors['wave_min'])  * np.random.uniform(), 
                                priors['wave_min']  + (priors['wave_max']   - priors['wave_min'])  * np.random.uniform(),   
                                priors['c0_1_min']  + (priors['c0_1_max']   - priors['c0_1_min'])  * np.random.uniform(), 
                                priors['c0_2_min']  + (priors['c0_2_max']   - priors['c0_2_min'])  * np.random.uniform(), 
                                priors['c1_1_min']  + (priors['c1_1_max']   - priors['c1_1_min'])  * np.random.uniform(), 
                                priors['c1_2_min']  + (priors['c1_2_max']   - priors['c1_2_min'])  * np.random.uniform(), 
                                priors['c2_1_min']  + (priors['c2_1_max']   - priors['c2_1_min'])  * np.random.uniform(), 
                                priors['c2_2_min']  + (priors['c2_2_max']   - priors['c2_2_min'])  * np.random.uniform(),
                                priors['N_min']     + (priors['N_max']      - priors['N_min'])     * np.random.uniform()
                            ]) for i in range(nwalkers)]


            ## multiprocessing
            print('Starting emcee')
            if Multiprocess:
                with Pool() as pool:
                    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf, xlsf), a=moves, pool=pool)
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf, xlsf), pool=pool, moves=emcee.moves.KDEMove())
                    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves)
                    time1 = time.time()
                    sampler.run_mcmc(pos, step, progress=True)
                    #print("Autocorrelation time: {0:.2f} steps".format(sampler.get_autocorr_time()[0]))
                    time2 = time.time()
            else:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf, xlsf), moves=emcee.moves.KDEMove())
                time1 = time.time()
                sampler.run_mcmc(pos, step, progress=True)
                time2 = time.time()
            print('Done with emcee')

            np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])
            samples = sampler.chain[:, :, :].reshape((-1, ndim))
            np.save(save_to_path + '/samples', samples)

            print('total time: ',(time2-time1)/60,' min.')
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print(sampler.acceptance_fraction)






        # create walker plots
        sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
        samples = np.load(save_to_path + '/samples.npy')

        ylabels = [
                   "$T_{eff} (K)$",
                   "$log \, g$",
                   "$[M/H]$",
                   "$vsin \, i (km/s)$",
                   "$RV (km/s)$",
                   "$AM$",
                   "$PWV$",
                   "$Wave Offset_1$",
                   "$Wave Offset_2$",
                   "$Wave Offset_3$",
                   "$C0_{1flux}$",
                   "$C0_{2flux}$",
                   "$C1_{1flux}$",
                   "$C1_{2flux}$",
                   "$C2_{1flux}$",
                   "$C2_{2flux}$",
                   "$C_{noise}$"
                   ]

        ## create walker plots
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=8)
        fig = plt.figure(figsize=(10,15), tight_layout=True)
        gs  = gridspec.GridSpec(ndim, 1)
        gs.update(hspace=0.1)

        for i in range(ndim):
            ax = fig.add_subplot(gs[i, :])
            for j in range(nwalkers):
                ax.plot(np.arange(1,int(step+1)), sampler_chain[j,:,i],'k',alpha=0.1)
                ax.set_ylabel(ylabels[i], fontsize=8)
        fig.align_labels()
        plt.minorticks_on()
        plt.xlabel('nstep', fontsize=8)
        plt.xlabel
        plt.savefig(save_to_path+'/walker.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        # create array triangle plots
        triangle_samples = sampler_chain[:, burn:, :].reshape((-1, ndim))
        #print(triangle_samples.shape)

        # create the final spectra comparison
        teff_mcmc, logg_mcmc, mh_mcmc, vsini_mcmc, rv_mcmc, am_mcmc, pwv_mcmc, wave1_mcmc, wave2_mcmc, wave3_mcmc, c0_1_mcmc, c0_2_mcmc, c1_1_mcmc, c1_2_mcmc, c2_1_mcmc, c2_2_mcmc, N_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
            zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

        # add the summary to the txt file
        log_path = save_to_path + '/mcmc_parameters.txt'
        file_log = open(log_path,"a")
        file_log.write("Total runtime (minutes): %0.2f \n"%((time2-time1)/60.))
        file_log.write("*** Below is the summary *** \n")
        #file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
        #file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
        file_log.write("teff_mcmc {} K\n".format(str(teff_mcmc)))
        file_log.write("logg_mcmc {} dex (cgs)\n".format(str(logg_mcmc)))
        file_log.write("mh_mcmc {} dex (cgs)\n".format(str(mh_mcmc)))
        file_log.write("vsini_mcmc {} km/s\n".format(str(vsini_mcmc)))
        file_log.write("rv_mcmc {} km/s\n".format(str(rv_mcmc)))
        file_log.write("am_mcmc {}\n".format(str(am_mcmc)))
        file_log.write("pwv_mcmc {}\n".format(str(pwv_mcmc)))
        file_log.write("wave1_mcmc {}\n".format(str(wave1_mcmc)))
        file_log.write("wave2_mcmc {}\n".format(str(wave2_mcmc)))
        file_log.write("wave3_mcmc {}\n".format(str(wave3_mcmc)))
        file_log.write("c0_1_mcmc {}\n".format(str(c0_1_mcmc)))
        file_log.write("c0_2_mcmc {}\n".format(str(c0_2_mcmc)))
        file_log.write("c1_1_mcmc {}\n".format(str(c1_1_mcmc)))
        file_log.write("c1_2_mcmc {}\n".format(str(c1_2_mcmc)))
        file_log.write("c2_1_mcmc {}\n".format(str(c2_1_mcmc)))
        file_log.write("c2_2_mcmc {}\n".format(str(c2_2_mcmc)))
        file_log.write("N_mcmc {}\n".format(str(N_mcmc)))
        file_log.close()

        # log file
        log_path2 = save_to_path + '/mcmc_result.txt'
        file_log2 = open(log_path2,"w+")
        file_log2.write("teff_mcmc {}\n".format(str(teff_mcmc[0])))
        file_log2.write("logg_mcmc {}\n".format(str(logg_mcmc[0])))
        file_log2.write("mh_mcmc {}\n".format(str(mh_mcmc[0])))
        file_log2.write("vsini_mcmc {}\n".format(str(vsini_mcmc[0])))
        file_log2.write("rv_mcmc {}\n".format(str(rv_mcmc[0]+barycorr)))
        file_log2.write("am_mcmc {}\n".format(str(am_mcmc[0])))
        file_log2.write("pwv_mcmc {}\n".format(str(pwv_mcmc[0])))
        file_log2.write("wave1_mcmc {}\n".format(str(wave1_mcmc[0])))
        file_log2.write("wave2_mcmc {}\n".format(str(wave2_mcmc[0])))
        file_log2.write("wave3_mcmc {}\n".format(str(wave3_mcmc[0])))
        file_log2.write("c0_1_mcmc {}\n".format(str(c0_1_mcmc[0])))
        file_log2.write("c0_2_mcmc {}\n".format(str(c0_2_mcmc[0])))
        file_log2.write("c1_1_mcmc {}\n".format(str(c1_1_mcmc[0])))
        file_log2.write("c1_2_mcmc {}\n".format(str(c1_2_mcmc[0])))
        file_log2.write("c2_1_mcmc {}\n".format(str(c2_1_mcmc[0])))
        file_log2.write("c2_2_mcmc {}\n".format(str(c2_2_mcmc[0])))
        file_log2.write("N_mcmc {}\n".format(str(N_mcmc[0])))
        file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
        file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
        file_log2.write("mh_mcmc_e {}\n".format(str(max(abs(mh_mcmc[1]), abs(mh_mcmc[2])))))
        file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
        file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
        file_log2.write("am_mcmc_e {}\n".format(str(max(abs(am_mcmc[1]), abs(am_mcmc[2])))))
        file_log2.write("pwv_mcmc_e {}\n".format(str(max(abs(pwv_mcmc[1]), abs(pwv_mcmc[2])))))
        file_log2.write("wave1_mcmc_e {}\n".format(str(max(abs(wave1_mcmc[1]), abs(wave1_mcmc[2])))))
        file_log2.write("wave2_mcmc_e {}\n".format(str(max(abs(wave2_mcmc[1]), abs(wave2_mcmc[2])))))
        file_log2.write("wave3_mcmc_e {}\n".format(str(max(abs(wave3_mcmc[1]), abs(wave3_mcmc[2])))))
        file_log2.write("c0_1_mcmc_e {}\n".format(str(max(abs(c0_1_mcmc[1]), abs(c0_1_mcmc[2])))))
        file_log2.write("c0_2_mcmc_e {}\n".format(str(max(abs(c0_2_mcmc[1]), abs(c0_2_mcmc[2])))))
        file_log2.write("c1_1_mcmc_e {}\n".format(str(max(abs(c1_1_mcmc[1]), abs(c1_1_mcmc[2])))))
        file_log2.write("c1_2_mcmc_e {}\n".format(str(max(abs(c1_2_mcmc[1]), abs(c1_2_mcmc[2])))))
        file_log2.write("c2_1_mcmc_e {}\n".format(str(max(abs(c2_1_mcmc[1]), abs(c2_1_mcmc[2])))))
        file_log2.write("c2_2_mcmc_e {}\n".format(str(max(abs(c2_2_mcmc[1]), abs(c2_2_mcmc[2])))))
        file_log2.write("N_mcmc_e {}\n".format(str(max(abs(N_mcmc[1]), abs(N_mcmc[2])))))
        file_log2.close()


        triangle_samples[:,4] += barycorr

        ## triangular plots
        plt.rc('font', family='sans-serif')
        fig = corner.corner(triangle_samples, 
            labels=ylabels,
            truths=[teff_mcmc[0], 
            logg_mcmc[0], 
            mh_mcmc[0],
            vsini_mcmc[0], 
            rv_mcmc[0]+barycorr, 
            am_mcmc[0], 
            pwv_mcmc[0],
            wave1_mcmc[0],
            wave2_mcmc[0],
            wave3_mcmc[0],
            c0_1_mcmc[0],
            c0_2_mcmc[0],
            c1_1_mcmc[0],
            c1_2_mcmc[0],
            c2_1_mcmc[0],
            c2_2_mcmc[0],
            N_mcmc[0]],
            quantiles=[0.16, 0.84],
            label_kwargs={"fontsize": 12})
        plt.minorticks_on()
        fig.savefig(save_to_path+'/triangle.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        teff        = teff_mcmc[0]
        logg        = logg_mcmc[0]
        z           = mh_mcmc[0]
        vsini       = vsini_mcmc[0]
        rv          = rv_mcmc[0]
        am          = am_mcmc[0]
        pwv         = pwv_mcmc[0]
        wave_off1   = wave1_mcmc[0]
        wave_off2   = wave2_mcmc[0]
        wave_off3   = wave3_mcmc[0]
        c0_1        = c0_1_mcmc[0]
        c0_2        = c0_2_mcmc[0]
        c1_1        = c1_1_mcmc[0]
        c1_2        = c1_2_mcmc[0]
        c2_1        = c2_1_mcmc[0]
        c2_2        = c2_2_mcmc[0]
        N           = N_mcmc[0]

        ## new plotting model 
        ## read in a model
        model        = smart.Model(teff=teff, logg=logg, z=z, modelset=modelset, instrument=instrument)

        # apply vmicro
        vmicro = 2.478 - 0.325*logg
        model.flux = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vmicro, rotate=False, gaussian=True)

        # apply vsini
        model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True)

        # apply rv (including the barycentric correction)
        model.wave   = smart.rvShift(model.wave, rv=rv)


        model_notell = copy.deepcopy(model)

        # apply telluric
        model        = smart.applyTelluric(model=model, airmass=am, pwv=pwv)


        ## APOGEE LSF
        model.flux = ap.apogee_hack.spec.lsf.convolve(model.wave, model.flux, lsf=lsf, xlsf=xlsf).flatten()
        model.wave = ap.apogee_hack.spec.lsf.apStarWavegrid()
        model_notell.flux = ap.apogee_hack.spec.lsf.convolve(model_notell.wave, model_notell.flux, lsf=lsf, xlsf=xlsf).flatten()
        model_notell.wave = ap.apogee_hack.spec.lsf.apStarWavegrid()
        # Remove the NANs
        model.wave = model.wave[~np.isnan(model.flux)]
        model.flux = model.flux[~np.isnan(model.flux)]
        model_notell.wave = model_notell.wave[~np.isnan(model_notell.flux)]
        model_notell.flux = model_notell.flux[~np.isnan(model_notell.flux)]

        # wavelength offset
        #model.wave += wave_offset

        # integral resampling
        #model.flux   = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
        #model.wave   = data.wave

        # contunuum correction
        #model, cont_factor = smart.continuum(data=data, mdl=model, prop=True)

        deg         = 5

        ## because of the APOGEE bands, continuum is corrected from three pieces of the spectra
        data0       = copy.deepcopy(data)
        model0      = copy.deepcopy(model)
        
        # wavelength offset
        model0.wave += wave_off1

        range0      = np.where((data0.wave >= data.oriWave0[0][-1]) & (data0.wave <= data.oriWave0[0][0]))
        data0.wave  = data0.wave[range0]
        data0.flux  = data0.flux[range0]
        model0.flux = np.array(smart.integralResample(xh=model0.wave, yh=model0.flux, xl=data0.wave))
        model0.wave = data0.wave
        model0, cont_factor0, constA0, constB0 = smart.continuum(data=data0, mdl=model0, deg=deg, prop=True)
                    
        data1       = copy.deepcopy(data)
        model1      = copy.deepcopy(model)
        
        # wavelength offset
        model1.wave += wave_off2

        range1      = np.where((data1.wave >= data.oriWave0[1][-1]) & (data1.wave <= data.oriWave0[1][0]))
        data1.wave  = data1.wave[range1]
        data1.flux  = data1.flux[range1]
        model1.flux = np.array(smart.integralResample(xh=model1.wave, yh=model1.flux, xl=data1.wave))
        model1.wave = data1.wave
        model1, cont_factor1, constA1, constB1 = smart.continuum(data=data1, mdl=model1, deg=deg, prop=True)
                    
        data2       = copy.deepcopy(data)
        model2      = copy.deepcopy(model)

        # wavelength offset
        model2.wave += wave_off3

        range2      = np.where((data2.wave >= data.oriWave0[2][-1]) & (data2.wave <= data.oriWave0[2][0]))
        data2.wave  = data2.wave[range2]
        data2.flux  = data2.flux[range2]
        model2.flux = np.array(smart.integralResample(xh=model2.wave, yh=model2.flux, xl=data2.wave))
        model2.wave = data2.wave
        model2, cont_factor2, constA2, constB2 = smart.continuum(data=data2, mdl=model2, deg=deg, prop=True)

        ## continuum for model w/o telluric
        model_notell0      = copy.deepcopy(model_notell)
        model_notell0.flux = np.array(smart.integralResample(xh=model_notell0.wave, yh=model_notell0.flux, xl=data0.wave))
        model_notell0.wave = data0.wave
        model_notell0.flux *= cont_factor0
        model_notell0.flux *= constA0
        model_notell0.flux -= constB0
                    
        model_notell1      = copy.deepcopy(model_notell)
        model_notell1.flux = np.array(smart.integralResample(xh=model_notell1.wave, yh=model_notell1.flux, xl=data1.wave))
        model_notell1.wave = data1.wave
        model_notell1.flux *= cont_factor1
        model_notell1.flux *= constA1
        model_notell1.flux -= constB1

        model_notell2      = copy.deepcopy(model_notell)
        model_notell2.flux = np.array(smart.integralResample(xh=model_notell2.wave, yh=model_notell2.flux, xl=data2.wave))
        model_notell2.wave = data2.wave
        model_notell2.flux *= cont_factor2
        model_notell2.flux *= constA2
        model_notell2.flux -= constB2

        # flux offset
        model0.flux        = (model0.flux + c0_1) * np.e**(-c0_2)
        model_notell0.flux = (model_notell0.flux + c0_1) * np.e**(-c0_2)
        model1.flux        = (model1.flux + c1_1) * np.e**(-c1_2)
        model_notell1.flux = (model_notell1.flux + c1_1) * np.e**(-c1_2)
        model2.flux        = (model2.flux + c2_1) * np.e**(-c2_2)
        model_notell2.flux = (model_notell2.flux + c2_1) * np.e**(-c2_2)

        model_notell.flux  = np.array( list(model_notell2.flux) + list(model_notell1.flux) + list(model_notell0.flux) )
        model_notell.wave  = np.array( list(model_notell2.wave) + list(model_notell1.wave) + list(model_notell0.wave) )

        model.flux  = np.array( list(model2.flux) + list(model1.flux) + list(model0.flux) )
        model.wave  = np.array( list(model2.wave) + list(model1.wave) + list(model0.wave) )



        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        #ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8)
        ax1.plot(model0.wave, model0.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model1.wave, model1.flux, color='C3', linestyle='-', label='',alpha=0.8, lw=lw)
        ax1.plot(model2.wave, model2.flux, color='C3', linestyle='-', label='',alpha=0.8, lw=lw)
        #ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8)
        ax1.plot(model_notell0.wave,model_notell0.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(model_notell1.wave,model_notell1.flux, color='C0', linestyle='-', label='',alpha=0.8, lw=lw)
        ax1.plot(model_notell2.wave,model_notell2.flux, color='C0', linestyle='-', label='',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-',label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ##### zoom-in plots for each chip
        ######################################################################################

        ######################################################################################
        ## larger figure
        ######################################################################################

        fig = plt.figure(figsize=(30,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data', alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_zoom.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ## chip c
        ######################################################################################

        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()
        plt.xlim(data.oriWave0[2][-1], data.oriWave0[2][-0])

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_chip_c.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ## chip b
        ######################################################################################

        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()
        plt.xlim(data.oriWave0[1][-1], data.oriWave0[1][-0])

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_chip_b.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ## chip a
        ######################################################################################

        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()
        plt.xlim(data.oriWave0[0][-1], data.oriWave0[0][-0])

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_chip_a.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()





        ###############################################################################################################################################


        # Get params
        teff0  = teff_mcmc[0]
        logg0  = logg_mcmc[0]
        mh0    = mh_mcmc[0]
        vsini0 = vsini_mcmc[0]
        rv0    = rv_mcmc[0]
        am0    = am_mcmc[0]
        pwv0   = pwv_mcmc[0]
        wave10 = wave1_mcmc[0]
        wave20 = wave2_mcmc[0]
        wave30 = wave3_mcmc[0]
        c0_10  = c0_1_mcmc[0]
        c0_20  = c0_2_mcmc[0]
        c1_10  = c1_1_mcmc[0]
        c1_20  = c1_2_mcmc[0]
        c2_10  = c2_1_mcmc[0]
        c2_20  = c2_2_mcmc[0]
        N0     = N_mcmc[0]


        ############################################



        BadPixMask  = pixels[np.where(abs(data.flux-model.flux) > np.median(data.flux-model.flux)+outlier_rejection*np.std(data.flux-model.flux))] 
        print('Bad Pix:', BadPixMask)

        # Mask out bad pixel mask
        pixels     = np.delete(pixels, BadPixMask)
        data.flux  = np.delete(data.flux, BadPixMask)
        data.noise = np.delete(data.noise, BadPixMask)
        data.wave  = np.delete(data.wave, BadPixMask)

        save_to_path           = save_BASE + '/apvisit/kounkel/' + source + '_visit-' + str(visit) + '/mcmc_APLSF_threeCont_threeWave_Vsini_Vmicro_KDEmove_c0c1_TelluricFit_3D_%s_%s_%s_badpix'%(modelset, nwalkers2,step2)

        if save_to_path is not None:
            if not os.path.exists(save_to_path):
                os.makedirs(save_to_path)
        else:
            save_to_path = '.'



        if MCMC:

            log_path = save_to_path + '/mcmc_parameters.txt'
            file_log = open(log_path,"w+")
            file_log.write("data_path {} \n".format(data.path))
            file_log.write("data_name {} \n".format(source))
            file_log.write("custom_mask {} \n".format(custom_mask))
            file_log.write("priors {} \n".format(priors))
            file_log.write("ndim {} \n".format(ndim2))
            file_log.write("nwalkers {} \n".format(nwalkers2))
            file_log.write("step {} \n".format(step2))
            file_log.write("burn {} \n".format(burn2))
            file_log.write("pixel_start {} \n".format(pixel_start))
            file_log.write("pixel_end {} \n".format(pixel_end))
            file_log.write("barycorr {} \n".format(barycorr))
            #file_log.write("med_snr {} \n".format(med_snr))
            file_log.close()


            ## read the input custom mask and priors
            lines          = open(save_to_path+'/mcmc_parameters.txt').read().splitlines()
            custom_mask    = json.loads(lines[2].split('custom_mask')[1])
            priors         = ast.literal_eval(lines[3].split('priors ')[1])

            pos2 = [np.array([   
                                teff0  + teff0  * Fraction*np.random.uniform(low=-1, high=1), # Teff 
                                logg0  + logg0  * Fraction*np.random.uniform(low=-1, high=1), # Logg 
                                mh0    + mh0    * Fraction*np.random.uniform(low=-1, high=1), # [M/H]
                                vsini0 + vsini0 * Fraction*np.random.uniform(low=-1, high=1), # vsini
                                rv0    + rv0    * Fraction*np.random.uniform(low=-1, high=1), # rv 
                                am0    + am0    * Fraction*np.random.uniform(low=-1, high=1), # airmass 
                                pwv0   + pwv0   * Fraction*np.random.uniform(low=-1, high=1), # pwv 
                                wave10 + wave10 * Fraction*np.random.uniform(low=-1, high=1), # wave offset 
                                wave20 + wave20 * Fraction*np.random.uniform(low=-1, high=1), # wave offset 
                                wave30 + wave30 * Fraction*np.random.uniform(low=-1, high=1), # wave offset 
                                c0_10  + c0_10  * Fraction*np.random.uniform(low=-1, high=1), # flux const 
                                c0_20  + c0_20  * Fraction*np.random.uniform(low=-1, high=1), # flux const 
                                c1_10  + c1_10  * Fraction*np.random.uniform(low=-1, high=1), # flux const 
                                c1_20  + c1_20  * Fraction*np.random.uniform(low=-1, high=1), # flux const 
                                c2_10  + c2_10  * Fraction*np.random.uniform(low=-1, high=1), # flux const 
                                c2_20  + c2_20  * Fraction*np.random.uniform(low=-1, high=1), # flux const 
                                N0     + N0     * Fraction*np.random.uniform(low=-1, high=1), # Noise 
                            ]) for i in range(nwalkers2)]


            ## multiprocessing
            with Pool() as pool:
                #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf, xlsf), a=moves, pool=pool)
                sampler = emcee.EnsembleSampler(nwalkers2, ndim2, lnprob, args=(data, lsf, xlsf), pool=pool, moves=emcee.moves.KDEMove())
                #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves)
                time1 = time.time()
                sampler.run_mcmc(pos2, step2, progress=True)
                time2 = time.time()

            np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])
            samples = sampler.chain[:, :, :].reshape((-1, ndim2))
            np.save(save_to_path + '/samples', samples)

            print('total time: ',(time2-time1)/60,' min.')
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print(sampler.acceptance_fraction)








        # create walker plots
        sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
        samples = np.load(save_to_path + '/samples.npy')


        ## create walker plots
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=8)
        fig = plt.figure(figsize=(10,10), tight_layout=True)
        gs  = gridspec.GridSpec(ndim2, 1)
        gs.update(hspace=0.1)

        for i in range(ndim2):
            ax = fig.add_subplot(gs[i, :])
            for j in range(nwalkers2):
                ax.plot(np.arange(1,int(step2+1)), sampler_chain[j,:,i],'k',alpha=0.1)
                ax.set_ylabel(ylabels[i], fontsize=8)
        fig.align_labels()
        plt.minorticks_on()
        plt.xlabel('nstep', fontsize=8)
        plt.savefig(save_to_path+'/walker.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        # create array triangle plots
        triangle_samples = sampler_chain[:, burn2:, :].reshape((-1, ndim2))
        #print(triangle_samples.shape)

        # create the final spectra comparison
        teff_mcmc, logg_mcmc, mh_mcmc, vsini_mcmc, rv_mcmc, am_mcmc, pwv_mcmc, wave1_mcmc, wave2_mcmc, wave3_mcmc, c0_1_mcmc, c0_2_mcmc, c1_1_mcmc, c1_2_mcmc, c2_1_mcmc, c2_2_mcmc, N_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
            zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

        # add the summary to the txt file
        log_path = save_to_path + '/mcmc_parameters.txt'
        file_log = open(log_path,"a")
        file_log.write("Total runtime (minutes): %0.2f \n"%((time2-time1)/60.))
        file_log.write("*** Below is the summary *** \n")
        #file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
        #file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
        file_log.write("teff_mcmc {} K\n".format(str(teff_mcmc)))
        file_log.write("logg_mcmc {} dex (cgs)\n".format(str(logg_mcmc)))
        file_log.write("mh_mcmc {} dex (cgs)\n".format(str(mh_mcmc)))
        file_log.write("vsini_mcmc {} km/s\n".format(str(vsini_mcmc)))
        file_log.write("rv_mcmc {} km/s\n".format(str(rv_mcmc)))
        file_log.write("am_mcmc {}\n".format(str(am_mcmc)))
        file_log.write("pwv_mcmc {}\n".format(str(pwv_mcmc)))
        file_log.write("wave1_mcmc {}\n".format(str(wave1_mcmc)))
        file_log.write("wave2_mcmc {}\n".format(str(wave2_mcmc)))
        file_log.write("wave3_mcmc {}\n".format(str(wave3_mcmc)))
        file_log.write("c0_1_mcmc {}\n".format(str(c0_1_mcmc)))
        file_log.write("c0_2_mcmc {}\n".format(str(c0_2_mcmc)))
        file_log.write("c1_1_mcmc {}\n".format(str(c1_1_mcmc)))
        file_log.write("c1_2_mcmc {}\n".format(str(c1_2_mcmc)))
        file_log.write("c2_1_mcmc {}\n".format(str(c2_1_mcmc)))
        file_log.write("c2_2_mcmc {}\n".format(str(c2_2_mcmc)))
        file_log.write("N_mcmc {}\n".format(str(N_mcmc)))
        file_log.close()

        # log file
        log_path2 = save_to_path + '/mcmc_result.txt'
        file_log2 = open(log_path2,"w+")
        file_log2.write("teff_mcmc {}\n".format(str(teff_mcmc[0])))
        file_log2.write("logg_mcmc {}\n".format(str(logg_mcmc[0])))
        file_log2.write("mh_mcmc {}\n".format(str(mh_mcmc[0])))
        file_log2.write("vsini_mcmc {}\n".format(str(vsini_mcmc[0])))
        file_log2.write("rv_mcmc {}\n".format(str(rv_mcmc[0]+barycorr)))
        file_log2.write("am_mcmc {}\n".format(str(am_mcmc[0])))
        file_log2.write("pwv_mcmc {}\n".format(str(pwv_mcmc[0])))
        file_log2.write("wave1_mcmc {}\n".format(str(wave1_mcmc[0])))
        file_log2.write("wave2_mcmc {}\n".format(str(wave2_mcmc[0])))
        file_log2.write("wave3_mcmc {}\n".format(str(wave3_mcmc[0])))
        file_log2.write("c0_1_mcmc {}\n".format(str(c0_1_mcmc[0])))
        file_log2.write("c0_2_mcmc {}\n".format(str(c0_2_mcmc[0])))
        file_log2.write("c1_1_mcmc {}\n".format(str(c1_1_mcmc[0])))
        file_log2.write("c1_2_mcmc {}\n".format(str(c1_2_mcmc[0])))
        file_log2.write("c2_1_mcmc {}\n".format(str(c2_1_mcmc[0])))
        file_log2.write("c2_2_mcmc {}\n".format(str(c2_2_mcmc[0])))
        file_log2.write("N_mcmc {}\n".format(str(N_mcmc[0])))
        file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
        file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
        file_log2.write("mh_mcmc_e {}\n".format(str(max(abs(mh_mcmc[1]), abs(mh_mcmc[2])))))
        file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
        file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
        file_log2.write("am_mcmc_e {}\n".format(str(max(abs(am_mcmc[1]), abs(am_mcmc[2])))))
        file_log2.write("pwv_mcmc_e {}\n".format(str(max(abs(pwv_mcmc[1]), abs(pwv_mcmc[2])))))
        file_log2.write("c0_1_mcmc_e {}\n".format(str(max(abs(c0_1_mcmc[1]), abs(c0_1_mcmc[2])))))
        file_log2.write("c0_2_mcmc_e {}\n".format(str(max(abs(c0_2_mcmc[1]), abs(c0_2_mcmc[2])))))
        file_log2.write("c1_1_mcmc_e {}\n".format(str(max(abs(c1_1_mcmc[1]), abs(c1_1_mcmc[2])))))
        file_log2.write("c1_2_mcmc_e {}\n".format(str(max(abs(c1_2_mcmc[1]), abs(c1_2_mcmc[2])))))
        file_log2.write("c2_1_mcmc_e {}\n".format(str(max(abs(c2_1_mcmc[1]), abs(c2_1_mcmc[2])))))
        file_log2.write("c2_2_mcmc_e {}\n".format(str(max(abs(c2_2_mcmc[1]), abs(c2_2_mcmc[2])))))
        file_log2.write("N_mcmc_e {}\n".format(str(max(abs(N_mcmc[1]), abs(N_mcmc[2])))))
        file_log2.close()

        triangle_samples[:,3] += barycorr

        ## triangular plots
        plt.rc('font', family='sans-serif')
        fig = corner.corner(triangle_samples, 
            labels=ylabels,
            truths=[teff_mcmc[0], 
            logg_mcmc[0],
            mh_mcmc[0],
            vsini_mcmc[0], 
            rv_mcmc[0]+barycorr, 
            am_mcmc[0], 
            pwv_mcmc[0],
            wave1_mcmc[0],
            wave2_mcmc[0],
            wave3_mcmc[0],
            c0_1_mcmc[0],
            c0_2_mcmc[0],
            c1_1_mcmc[0],
            c1_2_mcmc[0],
            c2_1_mcmc[0],
            c2_2_mcmc[0],
            N_mcmc[0]],
            quantiles=[0.16, 0.84],
            label_kwargs={"fontsize": 12})
        plt.minorticks_on()
        fig.savefig(save_to_path+'/triangle.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        teff        = teff_mcmc[0]
        logg        = logg_mcmc[0]
        z           = mh_mcmc[0]
        vsini       = vsini_mcmc[0]
        rv          = rv_mcmc[0]
        am          = am_mcmc[0]
        pwv         = pwv_mcmc[0]
        wave_off1   = wave1_mcmc[0]
        wave_off2   = wave2_mcmc[0]
        wave_off3   = wave3_mcmc[0]
        c0_1        = c0_1_mcmc[0]
        c0_2        = c0_2_mcmc[0]
        c1_1        = c1_1_mcmc[0]
        c1_2        = c1_2_mcmc[0]
        c2_1        = c2_1_mcmc[0]
        c2_2        = c2_2_mcmc[0]
        N           = N_mcmc[0]

        ## new plotting model 
        ## read in a model
        model        = smart.Model(teff=teff, logg=logg, z=z, modelset=modelset, instrument=instrument)


        vmicro = 2.478 - 0.325*logg
        model.flux = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vmicro, rotate=False, gaussian=True)

        # apply vsini
        model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True)

        # apply rv (including the barycentric correction)
        model.wave   = smart.rvShift(model.wave, rv=rv)


        model_notell = copy.deepcopy(model)

        # apply telluric
        model        = smart.applyTelluric(model=model, airmass=am, pwv=pwv)

        ## APOGEE LSF
        model.flux = ap.apogee_hack.spec.lsf.convolve(model.wave, model.flux, lsf=lsf, xlsf=xlsf).flatten()
        model.wave = ap.apogee_hack.spec.lsf.apStarWavegrid()
        model_notell.flux = ap.apogee_hack.spec.lsf.convolve(model_notell.wave, model_notell.flux, lsf=lsf, xlsf=xlsf).flatten()
        model_notell.wave = ap.apogee_hack.spec.lsf.apStarWavegrid()
        # Remove the NANs
        model.wave = model.wave[~np.isnan(model.flux)]
        model.flux = model.flux[~np.isnan(model.flux)]
        model_notell.wave = model_notell.wave[~np.isnan(model_notell.flux)]
        model_notell.flux = model_notell.flux[~np.isnan(model_notell.flux)]

        # wavelength offset
        #model.wave += wave_offset

        # integral resampling
        #model.flux   = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
        #model.wave   = data.wave

        # contunuum correction
        #model, cont_factor = smart.continuum(data=data, mdl=model, prop=True)

        deg         = 5

        ## because of the APOGEE bands, continuum is corrected from three pieces of the spectra
        data0       = copy.deepcopy(data)
        model0      = copy.deepcopy(model)

        # wavelength offset
        model0.wave += wave_off1

        range0      = np.where((data0.wave >= data.oriWave0[0][-1]) & (data0.wave <= data.oriWave0[0][0]))
        data0.wave  = data0.wave[range0]
        data0.flux  = data0.flux[range0]
        model0.flux = np.array(smart.integralResample(xh=model0.wave, yh=model0.flux, xl=data0.wave))
        model0.wave = data0.wave
        model0, cont_factor0, constA0, constB0 = smart.continuum(data=data0, mdl=model0, deg=deg, prop=True)
                    
        data1       = copy.deepcopy(data)
        model1      = copy.deepcopy(model)

        # wavelength offset
        model1.wave += wave_off2

        range1      = np.where((data1.wave >= data.oriWave0[1][-1]) & (data1.wave <= data.oriWave0[1][0]))
        data1.wave  = data1.wave[range1]
        data1.flux  = data1.flux[range1]
        model1.flux = np.array(smart.integralResample(xh=model1.wave, yh=model1.flux, xl=data1.wave))
        model1.wave = data1.wave
        model1, cont_factor1, constA1, constB1 = smart.continuum(data=data1, mdl=model1, deg=deg, prop=True)
                    
        data2       = copy.deepcopy(data)
        model2      = copy.deepcopy(model)

        # wavelength offset
        model2.wave += wave_off3

        range2      = np.where((data2.wave >= data.oriWave0[2][-1]) & (data2.wave <= data.oriWave0[2][0]))
        data2.wave  = data2.wave[range2]
        data2.flux  = data2.flux[range2]
        model2.flux = np.array(smart.integralResample(xh=model2.wave, yh=model2.flux, xl=data2.wave))
        model2.wave = data2.wave
        model2, cont_factor2, constA2, constB2 = smart.continuum(data=data2, mdl=model2, deg=deg, prop=True)

        ## continuum for model w/o telluric
        model_notell0      = copy.deepcopy(model_notell)
        model_notell0.flux = np.array(smart.integralResample(xh=model_notell0.wave, yh=model_notell0.flux, xl=data0.wave))
        model_notell0.wave = data0.wave
        model_notell0.flux *= cont_factor0
        model_notell0.flux *= constA0
        model_notell0.flux -= constB0
                    
        model_notell1      = copy.deepcopy(model_notell)
        model_notell1.flux = np.array(smart.integralResample(xh=model_notell1.wave, yh=model_notell1.flux, xl=data1.wave))
        model_notell1.wave = data1.wave
        model_notell1.flux *= cont_factor1
        model_notell1.flux *= constA1
        model_notell1.flux -= constB1

        model_notell2      = copy.deepcopy(model_notell)
        model_notell2.flux = np.array(smart.integralResample(xh=model_notell2.wave, yh=model_notell2.flux, xl=data2.wave))
        model_notell2.wave = data2.wave
        model_notell2.flux *= cont_factor2
        model_notell2.flux *= constA2
        model_notell2.flux -= constB2

        # flux offset
        model0.flux        = (model0.flux + c0_1) * np.e**(-c0_2)
        model_notell0.flux = (model_notell0.flux + c0_1) * np.e**(-c0_2)
        model1.flux        = (model1.flux + c1_1) * np.e**(-c1_2)
        model_notell1.flux = (model_notell1.flux + c1_1) * np.e**(-c1_2)
        model2.flux        = (model2.flux + c2_1) * np.e**(-c2_2)
        model_notell2.flux = (model_notell2.flux + c2_1) * np.e**(-c2_2)

        model_notell.flux  = np.array( list(model_notell2.flux) + list(model_notell1.flux) + list(model_notell0.flux) )
        model_notell.wave  = np.array( list(model_notell2.wave) + list(model_notell1.wave) + list(model_notell0.wave) )

        model.flux  = np.array( list(model2.flux) + list(model1.flux) + list(model0.flux) )
        model.wave  = np.array( list(model2.wave) + list(model1.wave) + list(model0.wave) )



        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        #ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8)
        ax1.plot(model0.wave, model0.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model1.wave, model1.flux, color='C3', linestyle='-', label='',alpha=0.8, lw=lw)
        ax1.plot(model2.wave, model2.flux, color='C3', linestyle='-', label='',alpha=0.8, lw=lw)
        #ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8)
        ax1.plot(model_notell0.wave,model_notell0.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(model_notell1.wave,model_notell1.flux, color='C0', linestyle='-', label='',alpha=0.8, lw=lw)
        ax1.plot(model_notell2.wave,model_notell2.flux, color='C0', linestyle='-', label='',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-',label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ##### zoom-in plots for each chip
        ######################################################################################

        ######################################################################################
        ## larger figure
        ######################################################################################

        fig = plt.figure(figsize=(30,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_zoom.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ## chip c
        ######################################################################################

        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()
        plt.xlim(data.oriWave0[2][-1], data.oriWave0[2][-0])

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_chip_c.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ## chip b
        ######################################################################################

        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()
        plt.xlim(data.oriWave0[1][-1], data.oriWave0[1][-0])

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_chip_b.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()

        ######################################################################################
        ## chip a
        ######################################################################################

        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(111)
        plt.rc('font', family='sans-serif')
        plt.tick_params(labelsize=15)
        ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8, lw=lw)
        ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8, lw=lw)
        ax1.plot(data.wave,data.flux,'k-', label='data',alpha=0.5, lw=lw)
        ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8, lw=lw)
        plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
        plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
        plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
        plt.ylabel("Flux ($10^{-17} erg/s/cm^2/\AA$)",fontsize=15)
        plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
        plt.figtext(0.89,0.85,str(data.header['OBJID']),
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=15)
        plt.figtext(0.89,0.82,"$Teff \, {}^{{+{}}}_{{-{}}}/ logg \, {}^{{+{}}}_{{-{}}}/ [M/H] \, {}^{{+{}}}_{{-{}}}/ vsini \, {}^{{+{}}}_{{-{}}}/ RV \, {}^{{+{}}}_{{-{}}}$".format(\
            round(teff_mcmc[0]),
            round(teff_mcmc[1]),
            round(teff_mcmc[2]),
            round(logg_mcmc[0],2),
            round(logg_mcmc[1],2),
            round(logg_mcmc[2],2),
            round(mh_mcmc[0],2),
            round(mh_mcmc[1],2),
            round(mh_mcmc[2],2),
            round(vsini_mcmc[0],2),
            round(vsini_mcmc[1],2),
            round(vsini_mcmc[2],2),
            round(rv_mcmc[0]+barycorr,2),
            round(rv_mcmc[1],2),
            round(rv_mcmc[2],2)),
            color='C0',
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12)
        plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
            round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
        color='k',
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
        plt.minorticks_on()
        plt.xlim(data.oriWave0[0][-1], data.oriWave0[0][-0])

        """
        ax2 = ax1.twiny()
        ax2.plot(pixel, data.flux, color='w', alpha=0)
        ax2.set_xlabel('Pixel',fontsize=15)
        ax2.tick_params(labelsize=15)
        ax2.set_xlim(pixel[0], pixel[-1])
        ax2.minorticks_on()
        """

        #plt.legend()
        plt.savefig(save_to_path + '/spectrum_chip_a.png', dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        plt.close()
        plt.close('all')

        gc.collect() # Collect the garbage!
        #sys.exit()




