import smart
import numpy as np
import apogee_tools as ap
import matplotlib.pyplot as plt
from astropy.io import fits

apogee_id = '2M05351259-0523440'
prefix = '/home/l3wei/ONC/starrynight/data/APOGEE/'
object_path = prefix + apogee_id + '/'
apstar_path  = f'{object_path}specs/apStar-{apogee_id}.fits'
apvisit_path  = f'{object_path}specs/apVisit-{apogee_id}-1.fits'
modelset = 'phoenix-aces-agss-cond-2011'
instrument = 'apogee'
order = 'all'

hdul = fits.open(f'/home/l3wei/ONC/starrynight/data/APOGEE/{apogee_id}/specs/apStar-{apogee_id}.fits')
apstar = smart.Spectrum(name=apogee_id, path=apstar_path, instrument=instrument, apply_sigma_mask=True, datatype='apstar', applytell=True)
apvisit = smart.Spectrum(name=apogee_id, path=apvisit_path, instrument=instrument, apply_sigma_mask=True, datatype='apvisit', applytell=True)

with open(object_path + 'lsf.npy', 'rb') as file:
    xlsf = np.load(file)
    lsf = np.load(file)

teff=3500
logg=4


fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(apvisit.wave, apvisit.flux, color='C0', lw=1)
ax.vlines([apvisit.oriWave0[0][0], apvisit.oriWave0[0][-1]], ymin=min(apvisit.flux), ymax=max(apvisit.flux), colors=['C1', 'C2'], ls='dashed')
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(apstar.wave, apstar.flux, color='C0', lw=1)
ax.vlines([apstar.oriWave0[0][0], apstar.oriWave0[0][-1]], ymin=min(apvisit.flux), ymax=max(apvisit.flux), colors=['C1', 'C2'], ls='dashed')
plt.show()


apvisit_model = smart.makeModel(
    teff=teff, logg=logg, instrument=instrument, order=order, modelset=modelset, 
    wave_off1=0, wave_off2=0, wave_off3=0,
    c0_1=0, c0_2=0, c1_1=0, c1_2=0, c2_1=0, c2_2=0, 
    data=apvisit, lsf=lsf, xlsf=xlsf
)

apstar_model = smart.makeModel(
    teff=teff, logg=logg, instrument=instrument, order=order, modelset=modelset, 
    wave_off1=0, wave_off2=0, wave_off3=0,
    c0_1=0, c0_2=0, c1_1=0, c1_2=0, c2_1=0, c2_2=0, 
    data=apstar, lsf=lsf, xlsf=xlsf
)