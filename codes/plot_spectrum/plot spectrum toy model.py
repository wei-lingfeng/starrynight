import smart
import matplotlib.pyplot as plt 
import numpy as np 

date = (22, 1, 20)

sci_frame = 51
tel_frame = 28
name = 177
order = 33

Year = str(date[0]).zfill(2)
Month = str(date[1]).zfill(2)
Date = str(date[2]).zfill(2)

Month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
common_prefix = '/home/l3wei/ONC/data/NIRSPAO/20' + Year + Month_list[int(Month) - 1] + Date + '/reduced/'

if int(Year) > 18:
    # For data after 2018, sci_names = [nspec200118_0027, ...]
    sci_name = 'nspec' + Year + Month + Date + '_' + str(sci_frame).zfill(4)
    tel_name = 'nspec' + Year + Month + Date + '_' + str(tel_frame).zfill(4)
    BadPixMask  = np.concatenate([np.arange(0,20), np.arange(2000,2048)])

else:
    # For data prior to 2018 (2018 included)
    sci_name = Month_list[int(Month) - 1] + Date + 's' + str(sci_frame).zfill(4)
    tel_name = Month_list[int(Month) - 1] + Date + 's' + str(tel_frame).zfill(4)
    BadPixMask  = np.concatenate([np.arange(0,10), np.arange(1000,1024)])

sci_spec = smart.Spectrum(name=sci_name, order=order, path=common_prefix + 'nsdrp_out/fits/all')
tel_spec = smart.Spectrum(name=tel_name + '_calibrated', order=order, path=common_prefix + tel_name + '/O%s' %order)

# update the wavelength solution
sci_spec.updateWaveSol(tel_spec)

pixel = np.arange(len(sci_spec.flux))

# Automatically Mask out bad pixels: flux < 0
BadPixMask_auto = np.concatenate([BadPixMask, pixel[np.where(sci_spec.flux < 0)]])

pixel          = np.delete(pixel, BadPixMask_auto)
sci_spec.flux  = np.delete(sci_spec.flux, BadPixMask_auto)
sci_spec.noise = np.delete(sci_spec.noise, BadPixMask_auto)
sci_spec.wave  = np.delete(sci_spec.wave, BadPixMask_auto)


# Model Spectrums:
# mcmc parameters
mcmc_path = '/home/l3wei/ONC/data/NIRSPAO/20{}{}{}/reduced/mcmc_median/{}_O{}_params/MCMC_Params.txt'.format(Year, Month_list[int(Month) - 1], Date, name, [32, 33])

with open(mcmc_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    if line.startswith('teff:'):
        teff = float(line.strip('teff: \n').split(', ')[0])
    if line.startswith('vsini:'):
        vsini = float(line.strip('vsini: \n').split(', ')[0])
    if line.startswith('rv:'):
        rv = float(line.strip('rv: \n').split(', ')[0])
    if line.startswith('airmass:'):
        airmass = float(line.strip('airmass: \n').split(', ')[0])
    if line.startswith('pwv:'):
        pwv = float(line.strip('pwv: \n').split(', ')[0])
    if line.startswith('veiling:'):
        veiling = float(line.strip('veiling: \n').split(', ')[0])
    if line.startswith('lsf:'):
        lsf = float(line.strip('lsf: \n').split(', ')[0])
    if line.startswith('noise:'):
        noise = float(line.strip('noise: \n').split(', ')[0])
    if line.startswith('wave_offset_O32:'):
        wave_offset_O32 = float(line.strip('wave_offset_O32: \n').split(', ')[0])
    if line.startswith('flux_offset_O32:'):
        flux_offset_O32 = float(line.strip('flux_offset_O32: \n').split(', ')[0])
    if line.startswith('wave_offset_O33:'):
        wave_offset_O33 = float(line.strip('wave_offset_O33: \n').split(', ')[0])
    if line.startswith('flux_offset_O33:'):
        flux_offset_O33 = float(line.strip('flux_offset_O33: \n').split(', ')[0])

custom_teff = 4375
model, model_notel = smart.makeModel(custom_teff, order=order, data=sci_spec, logg=4, vsini=vsini, rv=0, airmass=airmass, pwv=pwv, veiling=veiling, lsf=lsf, wave_offset=wave_offset_O33, flux_offset=flux_offset_O33, z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)
# model_new, model_notel_new = smart.makeModel(custom_teff, order=order, data=sci_spec, logg=4, vsini=vsini, rv=rv, airmass=airmass, pwv=pwv, veiling=veiling, lsf=lsf, wave_offset=wave_offset_O33, flux_offset=flux_offset_O33, z=0, modelset='phoenix-aces-agss-cond-2011', output_stellar_model=True)

# Plot model + original spectrum:
fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
ax.plot(sci_spec.wave, sci_spec.flux, alpha=0.7, lw=0.7, label='data')
ax.plot(model.wave, model.flux, alpha=0.7, lw=0.7, label='model + telluric')
ax.plot(model_notel.wave, model_notel.flux, color='C3', alpha=0.7, lw=0.7, label='model')
ax.set_ylabel('Flux (counts/s)', fontsize=15)
ax.set_xlabel('$\lambda$ (\AA)', fontsize=15)
ax.minorticks_on()
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tel_spec.wave, tel_spec.flux, alpha=0.7, lw=0.7, label='telluric')
ax.plot(model.wave, model.flux - model_notel.flux, alpha=0.7, lw=0.7, label='telluric model')
ax.legend()
plt.show()

# # Compare original model notel with custom teff model notel:
# fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
# ax.plot(model_notel.wave, model_notel.flux, lw=0.7, label='{:.0f} K model no telluric'.format(teff))
# ax.plot(model_notel_new.wave, model_notel_new.flux, lw=0.7, label='{} K model no telluric'.format(custom_teff))
# ax.plot(model_notel_new.wave, (model_notel_new.flux - model_notel.flux), color='k', lw=0.7, label='difference')
# ax.set_ylabel('Flux (counts/s)', fontsize=15)
# ax.set_xlabel('$\lambda$ (\AA)', fontsize=15)
# ax.minorticks_on()
# ax.legend()
# plt.show()

# # Compare original model with custom teff model:
# fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
# ax.plot(model.wave, model.flux, alpha=0.7, lw=0.7, label='{:.0f} K model'.format(teff))
# ax.plot(model_new.wave, model_new.flux, alpha=0.7, lw=0.7, label='{} K model'.format(custom_teff))
# ax.plot(model_new.wave, (model_new.flux - model.flux), color='C7', alpha=0.7, lw=0.7, label='difference')
# ax.set_ylabel('Flux (counts/s)', fontsize=15)
# ax.set_xlabel('$\lambda$ (\AA)', fontsize=15)
# ax.minorticks_on()
# ax.legend()
# plt.show()




# # Plot original spectrums:
# fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
# # ax.plot(sci_spec.wave, sci_spec.flux, alpha=0.7, lw=0.7, label='data')
# # plt.axhline(y=np.median(sci_spec.flux) + 3.5*np.std(sci_spec.flux), linestyle='--', color='C1')
# ax.set_ylabel('Flux (counts/s)', fontsize=15)
# ax.set_xlabel('$\lambda$ (\AA)', fontsize=15)


# ax.plot(model_notel.wave, model_notel.flux, color='C0', label='original model', alpha=0.7, lw=0.7)
# ax.plot(model_notel_new.wave, model_notel_new.flux, color='C3', label='6000K model', alpha=0.7, lw=0.7)
# ax.minorticks_on()
# plt.show()


# # ax.plot(model.wave, model.flux, color='C3', label='model + telluric', alpha=0.7, lw=0.7)
# # ax.plot(model_new.wave, model_new.flux, color='C0', label='6000 K model', alpha=0.7, lw=0.7)
# # ax.plot(model.wave, model_new.flux - model.flux, 'C7', label='Difference')
# # ax.fill_between(sci_spec.wave, -sci_spec.noise, sci_spec.noise, facecolor='0.8', label='noise')
# # ax.plot(sci_spec.wave, sci_spec.flux-model.flux, 'k-', label='residual', alpha=0.7, lw=0.7)
# ax.minorticks_on()
# # ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(1, 0))
# plt.show()