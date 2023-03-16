import os, sys

path2nsdrp  = '/home/l3wei/packages/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py'

Year 	= 15
Month 	= 12
Day 	= 24

eta_frame  = 22

# A-B pairs.
frames = [
    [41, 40, 42, 43]
]

# standalone frames
standalone_frames = []

Aframes = []
Bframes = []

for object in frames:
    Aframes.extend([object[i] for i in range(len(object)) if i%4 in [0, 3]])
    Bframes.extend([object[i] for i in range(len(object)) if i%4 in [1, 2]])

# Aframes = [31, 34, 35]
# Bframes = [32, 33, 36]

Year 	= str(Year).zfill(2)
Month 	= str(Month).zfill(2)
Day 	= str(Day).zfill(2)
Month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

common_path = '/home/l3wei/ONC/Data/20' + Year + Month_list[int(Month)-1] + Day
name_prefix = 'nspec' + Year + Month + Day + '_'
flatfile    = common_path + '/reduced/MasterFlatNoDark.fits'
save_path	= common_path + '/reduced/nsdrp_out/'
etafile     = common_path + '/specs/' + name_prefix + str(eta_frame).zfill(4) + '.fits'
etalinefile = '/home/l3wei/packages/NIRSPEC-Data-Reduction-Pipeline/ir_etalonlines.dat'

# os.chdir(common_path + '/reduced')

for Aframe, Bframe in zip(Aframes, Bframes):
    Apath = common_path + '/specs/' + name_prefix + str(Aframe).zfill(4) + '.fits'
    Bpath = common_path + '/specs/' + name_prefix + str(Bframe).zfill(4) + '.fits'
    
    # Run the reduction pipeline
    os.system('python3 {} {} {} -b {} -out_dir={} -eta_filename={} -etalon_filename={} -spatial_jump_override -verbose -debug -dgn'.format(path2nsdrp, flatfile, Apath, Bpath, save_path, etafile, etalinefile))


# stand alone frames
for frame in standalone_frames:
    Apath = common_path + '/specs/' + name_prefix + str(frame).zfill(4) + '.fits'
    os.system('python3 {} {} {} -out_dir={} -eta_filename={} -etalon_filename={} -spatial_jump_override -verbose -debug -dgn'.format(path2nsdrp, flatfile, Apath, save_path, etafile, etalinefile))