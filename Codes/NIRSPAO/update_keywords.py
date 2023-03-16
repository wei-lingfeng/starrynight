import os

dates = [
    (15, 12, 23),
    (15, 12, 24),
    (16, 12, 14),
    (18, 2, 11),
    (18, 2, 12),
    (18, 2, 13),
    (19, 1, 12),
    (19, 1, 13),
    (19, 1, 16),
    (19, 1, 17),
    (20, 1, 18),
    (20, 1, 19),
    (20, 1, 20),
    (20, 1, 21),
    (21, 2, 1),
    (21, 10, 20),
    (22, 1, 18),
    (22, 1, 19),
    (22, 1, 20)
]

order = [34]

month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

for date in dates:
    year, month, day = date
    common_prefix = '/home/l3wei/ONC/Data/20{}{}{}/reduced/mcmc_median'.format(str(year).zfill(2), month_list[int(month) - 1], str(day).zfill(2))
    
    for path in [_[0] for _ in os.walk(common_prefix)]:
        if order==[32, 33]:
            ends = 'params'
        elif order==[34]:
            ends = 'logg'
        else:
            raise ValueError('Cannot find results of order {}.'.format(order))
        
        
        if path.endswith(ends):
            with open(path + '/MCMC_Params.txt', 'r') as file:
                raw = file.read()
            
            # line = raw[:17]
            # raw = raw.replace(line, line.replace(', ', ''))
            # raw = raw.replace('sci frames', 'sci_frames')
            # raw = raw.replace('tel frames', 'tel_frames')
            # raw = raw.replace('wave offset O32', 'wave_offset_O32')
            # raw = raw.replace('flux offset O32', 'flux_offset_O32')
            # raw = raw.replace('wave offset O33', 'wave_offset_O33')
            # raw = raw.replace('flux offset O33', 'flux_offset_O33')
            # raw = raw.replace('snr O32', 'snr_O32')
            # raw = raw.replace('snr O33', 'snr_O33')
            raw = raw.replace('[', '')
            raw = raw.replace(']', '')
            
            with open(path + '/MCMC_Params.txt', 'w') as file:
                file.writelines(raw)
            
            # os.rename(path, path.replace('O34_logg', 'O[34]_logg'))
        else:
            pass