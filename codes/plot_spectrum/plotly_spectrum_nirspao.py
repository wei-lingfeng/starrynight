import os
import copy
import smart
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import reduce

date = (15, 12, 23)
name = '322'
sci_frames = [75, 76, 77, 78]
tel_frames = [79, 80, 80, 79]
order = 33

year = str(date[0]).zfill(2)
month = str(date[1]).zfill(2)
day = str(date[2]).zfill(2)

month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
prefix = f'/home/l3wei/ONC/data/nirspao/20{year}{month_list[int(month) - 1]}{day}/reduced/'

sci_abba = []
tel_abba = []
for sci_frame, tel_frame in zip(sci_frames, tel_frames):
    
    if int(year) > 18:
        # For data after 2018, sci_names = [nspec200118_0027, ...]
        sci_name = 'nspec' + year + month + day + '_' + str(sci_frame).zfill(4)
        tel_name = 'nspec' + year + month + day + '_' + str(tel_frame).zfill(4)
        pixel_start = 20
        pixel_end = -48
    
    else:
        # For data prior to 2018 (2018 included)
        sci_name = month_list[int(month) - 1] + day + 's' + str(sci_frame).zfill(4)
        tel_name = month_list[int(month) - 1] + day + 's' + str(tel_frame).zfill(4)
        pixel_start = 10
        pixel_end = -30
    
    if name.endswith('A'):
        sci_spec = smart.Spectrum(name=f'{sci_name}_A', order=order, path=f'{prefix}extracted_binaries/{sci_name}/O{order}')
    elif name.endswith('B'):
        sci_spec = smart.Spectrum(name=f'{sci_name}_B', order=order, path=f'{prefix}extracted_binaries/{sci_name}/O{order}')
    else:
        sci_spec = smart.Spectrum(name=sci_name, order=order, path=f'{prefix}nsdrp_out/fits/all')
    
    sci_spec.pixel = np.arange(len(sci_spec.wave))
    sci_spec.snr = np.median(sci_spec.flux / sci_spec.noise)
    
    if os.path.exists(prefix + tel_name + '_defringe/O{}/'.format(order)):
        tel_name = tel_name + '_defringe'
    
    tel_spec = smart.Spectrum(name=f'{tel_name}_calibrated', order=order, path=f'{prefix}{tel_name}/O{order}/')
    
    # Update the wavelength solution
    sci_spec.updateWaveSol(tel_spec)
    
    # Automatically mask out edge & flux < 0
    mask1 = [True if (sci_spec.flux[i] < 0) or (i < pixel_start) or (i >= len(sci_spec.wave) + pixel_end) else False for i in np.arange(len(sci_spec.wave))]
    sci_spec.pixel  = ma.MaskedArray(sci_spec.pixel, mask=mask1)
    sci_spec.wave   = ma.MaskedArray(sci_spec.wave,  mask=mask1)
    sci_spec.flux   = ma.MaskedArray(sci_spec.flux,  mask=mask1)
    sci_spec.noise  = ma.MaskedArray(sci_spec.noise, mask=mask1)

    # Mask flux > median + 3 sigma
    median_flux = ma.median(sci_spec.flux)
    upper_bound = median_flux + 2.5*ma.std(sci_spec.flux - median_flux)
    mask2 = sci_spec.flux > upper_bound
    sci_spec.pixel  = ma.MaskedArray(sci_spec.pixel, mask=mask2)
    sci_spec.wave   = ma.MaskedArray(sci_spec.wave,  mask=mask2)
    sci_spec.flux   = ma.MaskedArray(sci_spec.flux,  mask=mask2)
    sci_spec.noise  = ma.MaskedArray(sci_spec.noise, mask=mask2)

    # Mask isolated bad pixels
    median_flux = ma.median(sci_spec.flux)
    lower_bound = median_flux - 3.5*ma.std(sci_spec.flux - median_flux)
    lowest_bound = median_flux - 5.*ma.std(sci_spec.flux - median_flux)
    mask3 = [False, *[True if (sci_spec.flux[i] < lowest_bound) and (sci_spec.flux[i-1] >= lower_bound) and (sci_spec.flux[i+1] >= lower_bound) else False for i in np.arange(1, len(sci_spec.wave)-1)], False]
    sci_spec.pixel  = ma.MaskedArray(sci_spec.pixel, mask=mask3)
    sci_spec.wave   = ma.MaskedArray(sci_spec.wave,  mask=mask3)
    sci_spec.flux   = ma.MaskedArray(sci_spec.flux,  mask=mask3)
    sci_spec.noise  = ma.MaskedArray(sci_spec.noise, mask=mask3)
    
    sci_abba.append(copy.deepcopy(sci_spec))
    tel_abba.append(copy.deepcopy(tel_spec))


# Normalize to the highest snr frame
median_flux = ma.median(sci_abba[np.argmax([_.snr for _ in sci_abba])].flux)
for spec in sci_abba:
    normalize_factor = median_flux / ma.median(spec.flux)
    spec.flux   *= normalize_factor
    spec.noise  *= normalize_factor

# tel_spec = tel_abba[np.argmin([_.header['RMS'] for _ in tel_abba])]

sci_spec = copy.deepcopy(sci_abba[np.argmin([_.header['RMS'] for _ in tel_abba])])
sci_spec.flux = ma.average(ma.array([_.flux for _ in sci_abba]), weights=1/ma.array([_.noise for _ in sci_abba])**2, axis=0)
sci_spec.noise = ma.sqrt(ma.std(ma.array([_.flux for _ in sci_abba]), axis=0)**2 + ma.sum(ma.array([_.noise for _ in sci_abba])**2, axis=0) / ma.sum(~ma.array([_.noise.mask for _ in sci_abba]), axis=0))
# sci_spec.noise = ma.sqrt(ma.sum(ma.array([_.noise for _ in sci_abba])**2, axis=0) / ma.sum(~ma.array([_.noise.mask for _ in sci_abba]), axis=0))
sci_spec.pixel.mask = sci_spec.flux.mask
sci_spec.wave.mask = sci_spec.flux.mask

sci_spec.pixel  = sci_spec.pixel.compressed()
sci_spec.wave   = sci_spec.wave.compressed()
sci_spec.flux   = sci_spec.flux.compressed()
sci_spec.noise  = sci_spec.noise.compressed()

# Special Case
if date == (19, 1, 12) and order == 32:
    idx = sci_spec.wave < 23980
    sci_spec.pixel  = sci_spec.pixel[idx]
    sci_spec.flux   = sci_spec.flux [idx]
    sci_spec.noise  = sci_spec.noise[idx]
    sci_spec.wave   = sci_spec.wave [idx]

itime = sci_spec.header['ITIME']


fig_data = []
for i, spec in enumerate(sci_abba):
    spec.pixel  = spec.pixel.compressed()
    spec.flux   = spec.flux.compressed()
    spec.noise  = spec.noise.compressed()
    fig_data.append(go.Scatter(x=spec.pixel, y=spec.flux, mode='lines+markers', name=f'Frame {i+1}', line=dict(width=1, color='#7f7f7f'), marker=dict(size=3)))
    fig_data.append(go.Scatter(x=spec.pixel, y=spec.noise, mode='lines+markers', name=f'Noise {i+1}', line=dict(width=1, color='#7f7f7f'), marker=dict(size=3)))

fig_data.append(go.Scatter(x=sci_spec.pixel, y=sci_spec.flux, mode='lines+markers', name='Coadd Spectrum', line=dict(width=1, color='#1f77b4'), marker=dict(size=3)))
fig_data.append(go.Scatter(x=sci_spec.pixel, y=sci_spec.noise, mode='lines+markers', name='Coadd Noise', line=dict(width=1, color='#1f77b4'), marker=dict(size=3)))

fig = go.Figure()
fig.add_traces(fig_data)
fig.update_layout(width=1000, height=500, xaxis = dict(tickformat='000'))
fig.update_layout(xaxis_title='Pixel', yaxis_title='Flux')
fig.show()