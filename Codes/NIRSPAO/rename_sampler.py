import os

prefix = '/home/l3wei/ONC/Data/'
with os.scandir(prefix) as it:
    # entry: 2016dec14
    for entry in it:
        if entry.name == '.ipynb_checkpoints':
            continue
        mcmc_median_path = '{}{}/reduced/mcmc_median/'.format(prefix, entry.name)
        with os.scandir(mcmc_median_path) as it2:
            # entry2: 219_O[32, 33]_params
            for entry2 in it2:
                if os.path.exists('{}{}/sampler2.h5'.format(mcmc_median_path, entry2.name)):
                    os.replace('{}{}/sampler2.h5'.format(mcmc_median_path, entry2.name), '{}{}/sampler.h5'.format(mcmc_median_path, entry2.name))
                if os.path.exists('{}{}/sampler1.h5'.format(mcmc_median_path, entry2.name)):
                    os.remove('{}{}/sampler1.h5'.format(mcmc_median_path, entry2.name))
                if os.path.exists('{}{}/sampler2.h5'.format(mcmc_median_path, entry2.name)):
                    os.remove('{}{}/sampler2.h5'.format(mcmc_median_path, entry2.name))