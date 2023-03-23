import sys, os
from astropy.io import fits
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from astropy.table import Table

Year = 22
Month = 1
Day = 20

Year = str(Year).zfill(2)
Month = str(Month).zfill(2)
Day = str(Day).zfill(2)

file_prefix = 'nspec' + Year + Month + Day + '_'
Month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
common_path = '/home/l3wei/ONC/Data/20' + Year + Month_list[int(Month)-1] + Day + '/'


# flat file numbers
files        = np.arange(11, 21)
DARK         = False # Set this if you want to use a Master Dark file
reducedpath  = common_path + 'reduced/'

# Make the reduced directory if it doesn't exist
if not os.path.exists(reducedpath):
	os.makedirs(reducedpath)

# NIRSPEC upgrade
data_path = common_path + 'specs/defringeflat/'
hdr = fits.getheader(data_path + file_prefix + '{:0>4}_defringe.fits'.format(files[0]), ignore_missing_end=True, output_verify='silentfix')

if DARK:

	# Get the Dark
	darkData = fits.getdata(reducedpath+'MasterDark.fits', ignore_missing_end=True, output_verify='silentfix')

	# NIRSPEC upgrade
	flatData = np.array([fits.getdata(data_path + file_prefix + '{:0>4}_defringe.fits'.format(file), ignore_missing_end=True) - darkData for file in files])
	
	flat     = np.median(flatData, axis=0)
	fits.writeto(reducedpath + "MasterFlat.fits", flat, hdr, overwrite=True)
	print('output:', reducedpath + "MasterFlat.fits")

else: 

	# NIRSPEC upgrade
	flatData = np.array([fits.getdata(data_path + file_prefix + '{:0>4}_defringe.fits'.format(file), ignore_missing_end=True, output_verify='silentfix') for file in files])
	
	flat     = np.median(flatData, axis=0)
	fits.writeto(reducedpath + "MasterFlatNoDark.fits", flat, hdr, overwrite=True, output_verify='ignore')
	print('output:', reducedpath+"MasterFlatNoDark.fits")
