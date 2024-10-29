import os
from itertools import repeat
from nirspao_sources import nirspao_sources
from construct_synthetic_catalog import construct_synthetic_catalog

user_path = os.path.expanduser('~')

dates = [
    *list(repeat((15, 12, 23), 4)),
    *list(repeat((15, 12, 24), 8)),
    *list(repeat((16, 12, 14), 4)),
    *list(repeat((18, 2, 11), 7)),
    *list(repeat((18, 2, 12), 5)),
    *list(repeat((18, 2, 13), 6)),
    *list(repeat((19, 1, 12), 5)),
    *list(repeat((19, 1, 13), 6)),
    *list(repeat((19, 1, 16), 6)),
    *list(repeat((19, 1, 17), 5)),
    *list(repeat((20, 1, 18), 2)),
    *list(repeat((20, 1, 19), 3)),
    *list(repeat((20, 1, 20), 6)),
    *list(repeat((20, 1, 21), 7)),
    *list(repeat((21, 2, 1), 2)),
    *list(repeat((21, 10, 20), 4)),
    *list(repeat((22, 1, 18), 6)),
    *list(repeat((22, 1, 19), 5)),
    *list(repeat((22, 1, 20), 7))
]

names = [
    # 2015-12-23
    322, 296, 259, 213,
    # 2015-12-24
    '306_A', '306_B', '291_A', '291_B', 252, 250, 244, 261,
    # 2016-12-14
    248, 223, 219, 324,
    # 2018-2-11
    295, 313, 332, 331, 337, 375, 388,
    # 2018-2-12
    425, 713, 408, 410, 436,
    # 2018-2-13
    '354_B2', '354_B3', '354_B1', '354_B4', 442, 344,
    # 2019-1-12
    '522_A', '522_B', 145, 202, 188,
    # 2019-1-13
    302, 275, 245, 258, 220, 344,
    # 2019-1-16
    370, 389, 386, 398, 413, 253,
    # 2019-1-17
    288, 420, 412, 282, 217,
    # 2020-1-18
    217, 229,
    # 2020-1-19
    228, 224, 135,
    # 2020-1-20
    440, 450, 277, 204, 229, 214,
    # 2020-1-21
    215, 240, 546, 504, 703, 431, 229,
    # 2021-2-1
    484, 476,
    # 2021-10-20
    546, 217, 277, 435,
    # 2022-1-18
    457, 479, 490, 478, 456, 170, 
    # 2022-1-19
    453, 438, 530, 287, 171, 
    # 2022-1-20
    238, 266, 247, 172, 165, 177, 163
]

exceptions = {
    'dates':[
        (15, 12, 24),
        (18, 2, 11)
    ],
    'names':[
        '291_A',
        337
    ],
    'orders':[35]
}

# create nirspec sources catalog.
result = nirspao_sources(dates=dates, names=names, exceptions=exceptions, save_path=f'{user_path}/ONC/starrynight/catalogs/nirspec sources.csv')

nirspec_path        = f'{user_path}/ONC/starrynight/catalogs/nirspec sources.csv'
chris_table_path    = f'{user_path}/ONC/starrynight/catalogs/Chris\'s Table.csv'
apogee_path         = f'{user_path}/ONC/starrynight/catalogs/apogee x 2mass.csv'
kounkel_path        = f'{user_path}/ONC/starrynight/catalogs/kounkel 2018.csv'
pm_path             = f'{user_path}/ONC/starrynight/catalogs/proper motion.csv'
gaia_path           = f'{user_path}/ONC/starrynight/catalogs/gaia dr3.csv'
hillenbrand_path    = f'{user_path}/ONC/starrynight/catalogs/hillenbrand mass.csv'
tobin_path          = f'{user_path}/ONC/starrynight/catalogs/tobin 2009.csv'
save_path           = f'{user_path}/ONC/starrynight/catalogs/'

# create nirspec + apogee + gaia sources catalog.
sources, sources_epoch_combined = construct_synthetic_catalog(nirspec_path, chris_table_path, apogee_path, kounkel_path, pm_path, gaia_path, hillenbrand_path, tobin_path, save_path)