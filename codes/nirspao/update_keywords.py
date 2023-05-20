import os

user_path = os.path.expanduser('~')
dates = [
    # (15, 12, 23),
    # (15, 12, 24),
    # (16, 12, 14),
    # (18, 2, 11),
    # (18, 2, 12),
    # (18, 2, 13),
    # (19, 1, 12),
    # (19, 1, 13),
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

order = [32, 33]

month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

for date in dates:
    year, month, day = date
    common_prefix = f'{user_path}/ONC/data/nirspao/20{str(year).zfill(2)}{month_list[int(month) - 1]}{str(day).zfill(2)}/reduced/mcmc_median'
    
    for path in [_[0] for _ in os.walk(common_prefix)]:
        if order==[32, 33]:
            ends = 'params'
        elif order==[34]:
            ends = 'logg'
        else:
            raise ValueError('Cannot find results of order {}.'.format(order))
        
        
        if path.endswith('_'.join((str(order), ends))):
            with open(f'{path}/mcmc_params.txt', 'r') as file:
                raw = file.readlines()
            
            for i, line in enumerate(raw):
                if line.startswith('itime: \t'):
                    itime = int(eval(line.strip('itime: \t\n'))/1000)
                    raw[i] = f'itime: \t{itime}\n'
            
            with open(path + '/mcmc_params.txt', 'w') as file:
                file.writelines(raw)
            
            # os.rename(path, path.replace('O34_logg', 'O[34]_logg'))
        else:
            pass