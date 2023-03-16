import pandas as pd
from astropy.table import Table

# kounkel = Table.read('/home/l3wei/ONC/Catalogs/kounkel 2018.fit')
# df = kounkel.to_pandas()
# df.to_csv('/home/l3wei/ONC/Catalogs/kounkel 2018.csv', index=False)

tobin = Table.read('/home/l3wei/ONC/Catalogs/tobin 2009.fit')
df = tobin.to_pandas()
df.to_csv('/home/l3wei/ONC/Catalogs/tobin 2009.csv', index=False)