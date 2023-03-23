import os, sys, copy, gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smart
from smart.utils.mask import generate_telluric_mask

topbase = os.getenv("HOME")

master_file_path   = '../tables/telluric_calibration_master_L_dwarf.xlsx'
master_file_path2  = '../tables/telluric_calibration_master_L_dwarf_update.xlsx'
#save_BASE          = '/Volumes/LaCie/nirspec/tellurics/'
save_BASE          = topbase + '/nirspec/tellurics/'

test               = False
save               = True
applymask          = False
pixel_start        = 10 #10
pixel_end          = -20 #-80 for upgraded NIRSPEC

df = pd.read_excel(master_file_path)

#date_obs_prev = None

# handle a data type bug for pandas
df['tell_mask'] = df['tell_mask'].astype('object')

for i in range(len(df)):
	# skip the calibrated files
	print(i)
	if type(df['wavecal_error'][i]) is str:
		print('skip file {} on {}'.format(df['tell_name'][i], df['date_obs'][i])) 
		continue

	# reset the mask; old NIRSPEC
	mask0     = np.concatenate((np.arange(1024)[:pixel_start], np.arange(1024)[pixel_end:]), axis=0)
	mask      = [int(i) for i in mask0]
	mask0     = mask
	# reset pwv
	pwv       = '0.5'

	date_obs0 = df['date_obs'][i]
	a         = date_obs0.split('-')
	date_obs  = a[0]+a[1]+a[2]

	data_path = df['tell_path'][i]
	data_name = df['tell_name'][i]
	order     = int(df['order'][i])
	order_list = [order]

	## calibrate only the first four files under the same night
	#if date_obs_prev != date_obs:
	#	date_obs_count = 0
	#else:
	#	if date_obs_count >= 4:
	#		continue

	## masking selection
	"""
	data = smart.Spectrum(name=data_name, order=33, path=data_path)
	data2 = copy.deepcopy(data)
	length = len(data2.oriWave)
	pixel0 = np.arange(length)
	print(pixel0[np.where(data2.oriFlux > 275)])
	pixel      = np.delete(np.arange(length), mask)[pixel_start: pixel_end]
	#print(len(data2.oriFlux), len(data2.oriWave))
	data2.flux = np.delete(data2.oriFlux, mask)[pixel_start: pixel_end]
	data2.wave = np.delete(data2.oriWave, mask)[pixel_start: pixel_end]

	plt.figure(figsize=(16,6))
	plt.plot(np.arange(length), data.oriFlux,'k-',alpha=0.5)
	plt.plot(pixel, data2.flux,'r-',alpha=0.5)
	plt.ylabel('cnt/s')
	plt.xlabel('pixel')
	#plt.ylim(60, 250)
	plt.minorticks_on()
	plt.show()
	plt.close()
	sys.exit()
	"""

	save_to_path = save_BASE + date_obs + '/calibration/' + data_name

	print(save_to_path)
	print("Telluric wavelength calibration of {} on ".format(data_name, date_obs))
	smart.run_wave_cal(data_name, data_path, order_list,
		save_to_path, test=test, save=save, apply_sigma_mask=applymask, mask_custom=mask, pwv=pwv)

	# apply a telluric mask
	mask, pwv = generate_telluric_mask(name=data_name+'_calibrated', order=order, 
		path=save_to_path+'/O{}'.format(order_list[0]), 
		pixel_start=pixel_start, pixel_end=pixel_end, sigma=3.0, guess_pwv=True, diagnostic=True)

	mask = list(set(mask + mask0))
	mask.sort()

	# move the previous folder
	os.renames(save_to_path+'/O{}'.format(order_list[0]), save_to_path+'/O{}_last'.format(order))

	smart.run_wave_cal(data_name, data_path, order_list,
		save_to_path, test=test, save=save, apply_sigma_mask=applymask, mask_custom=mask, pwv=pwv)

	# logging
	data = smart.Spectrum(name=data_name+'_calibrated', order=order, path=save_to_path+'/O{}'.format(order))

	df['airmass'][i]       = data.header['AIRMASS']
	df['pwv'][i]           = pwv
	df.at[i, 'tell_mask']  = mask
	df['wavecal_error'][i] = data.header['RMS']

	df.to_excel(master_file_path2, index=False)

	## increase the date_obs count
	#date_obs_prev   = date_obs
	#date_obs_count += 1

	# clean the trash
	delete_param = [mask0, mask, pwv, date_obs0, date_obs, a, data_path, data_name, order, order_list, save_to_path, data]
	for param in delete_param:
		del param
	del delete_param

	gc.collect()

	#sys.exit()
