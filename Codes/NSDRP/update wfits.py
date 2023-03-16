from smart.utils.data_reduction_utils import update_wfits_param

year    = 15
month   = 12
day     = 24
names   = [51]
longer_sci = 56
orders = [35]

year = str(year).zfill(2)
month = str(month).zfill(2)
day = str(day).zfill(2)
longer_sci = str(longer_sci).zfill(4)

Month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
data_path = '/home/l3wei/ONC/Data/20' + year + Month_list[int(month)-1] + day + '/reduced/nsdrp_out/fits/all'

names = [str(i).zfill(4) for i in names]

if int(year) > 18:
    filename_B = 'nspec' + year + month + day + '_' + longer_sci
else:
    filename_B = Month_list[int(month)-1] + day + 's' + longer_sci

for name in names:
    for order in orders:
        if int(year) > 18:
            filename_A = 'nspec' + year + month + day + '_' + name
        else:
            filename_A = Month_list[int(month)-1] + day + 's' + name
        update_wfits_param(filename_A=filename_A, filename_B=filename_B, order=order, data_path=data_path, verbose=True)