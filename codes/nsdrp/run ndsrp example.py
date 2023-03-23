import sys, os

path2nsdrp  = '/home/l3wei/packages/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py'
flatfile    = '/home/l3wei/ONC/Data/2021feb01/reduced/MasterFlatNoDark.fits'
Aframe      = '/home/l3wei/ONC/Data/2021feb01/specs/nspec210201_0037.fits'
Bframe      = '/home/l3wei/ONC/Data/2021feb01/specs/nspec210201_0038.fits'
save_path	= '/home/l3wei/ONC/Data/2021feb01/reduced/'
etafile     = '/home/l3wei/ONC/Data/2021feb01/specs/nspec210201_0023.fits'
etalinefile = '/home/l3wei/packages/NIRSPEC-Data-Reduction-Pipeline/ir_etalonlines.dat'


# Run the reduction pipeline
os.system('python3 %s %s %s -b %s -out_dir=%s -eta_filename=%s -etalon_filename=%s -spatial_jump_override -verbose -debug -dgn'%(path2nsdrp,
	flatfile, Aframe, Bframe, save_path, etafile, etalinefile))

