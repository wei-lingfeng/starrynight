import os
import smart

year    = 18
month   = 2
day     = 11
frames  = [51, 52]
orders  = [36]

month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

user_path = os.path.expanduser('~')

for frame in frames:
    
    name = str(frame)
    data_path = f'{user_path}/ONC/Data/20{str(year).zfill(2)}{month_list[month-1]}{str(day).zfill(2)}/reduced/nsdrp_out/fits/all'
    
    if year > 18:
        data_name = f'nspec{str(year).zfill(2)}{str(month).zfill(2)}{str(day).zfill(2)}_{name.zfill(4)}' # Filename of the telluric standard
    else:
        data_name = f'{month_list[int(month)-1]}{str(day).zfill(2)}s{name.zfill(4)}'
    
    save_path = f'{user_path}/ONC/Data/20{str(year).zfill(2)}{month_list[month-1]}{str(day).zfill(2)}/reduced/data_name'
    
    smart.run_wave_cal(
        data_name=data_name,
        data_path=data_path,
        order_list=orders,
        save_to_path=save_path,
        save=True,
        plot_masked=True,
        apply_sigma_mask=True,
        apply_edge_mask=True,
        pwv='1.5'
    )