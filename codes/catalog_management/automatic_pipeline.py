from itertools import repeat
from nirspec_sources import nirspec_sources
from construct_synthetic_catalog import construct_synthetic_catalog

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
    *list(repeat((21, 10, 20), 4))
]

names = [
    # 2015-12-23
    322, 296, 259, 213,
    # 2015-12-24
    '306A', '306B', '291A', '291B', 252, 250, 244, 261,
    # 2016-12-14
    248, 223, 219, 324,
    # 2018-2-11
    295, 313, 332, 331, 337, 375, 388,
    # 2018-2-12
    425, 713, 408, 410, 436,
    # 2018-2-13
    '354B2B3_A', '354B2B3_B', '354B1', '354B4', 442, 344,
    # 2019-1-12
    '522A', '522B', 145, 202, 188,
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
    546, 217, 277, 435
]

# paths
nirspec_path    = '/home/l3wei/ONC/Catalogs/nirspec sources.csv'
apogee_path     = '/home/l3wei/ONC/Catalogs/apogee x 2mass.csv'
gaia_path       = '/home/l3wei/ONC/Catalogs/gaia sources.csv'
save_path       = '/home/l3wei/ONC/Catalogs/nirspec apogee gaia.csv'

# create nirspec sources catalog.
nirspec_sources(dates, names, save_path=nirspec_path)

# create nirspec + apogee + gaia sources catalog.
construct_synthetic_catalog(nirspec_path, apogee_path, gaia_path, save_path)