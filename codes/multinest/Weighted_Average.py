import os 
import numpy as np 
import matplotlib.pyplot as plt 
import csv 

def teff_avrg(year, month, dates, frames):
    '''
        Read and calculate the weighted average teffs of one month's observation from Multinest.\n
        year = 20\\
        month = 1\\
        dates = [18, 19, 20]\\
        frames[i][j][k] = i-th date, j-th object, k-th frame\\
        Returns 3*N array: teff_avrg[:, i] = [teff, lower, upper]
    '''
    orders = [32, 33]
    output_prefix = 'M_'

    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    year = str(year).zfill(2)
    month = str(month).zfill(2)
    teff_avrg = []

    for idx, date in enumerate(dates):
        date = str(date).zfill(2)
        
        for object in frames[idx]:
            teff_object = []
            
            for frame in object:
                
                if int(year) > 18:
                    sci_name = 'nspec' + year + month + date + '_' + str(frame).zfill(4)
                else:
                    sci_name = month_list[int(month) - 1] + date + 's' + str(frame).zfill(4)
                
                #########################################
                ############# Read Multinest ############
                #########################################
                teff_path = '/home/l3wei/ONC/Data/20' + year + month_list[int(month)-1] + date + '/reduced/multinest/' + sci_name + '_' + str(orders) + '/' + output_prefix + 'Multinest_Params.txt'
                
                with open(teff_path) as file:
                    teff_raw = file.readlines()
                    teff_temp = [float(i) for i in teff_raw[0].strip('teff: \t').split(', ')]
                    teff_object.append(teff_temp) 
            
            # Calculate weighted average:
            teff_object = np.array(teff_object)
            teff_weights = 1/np.power(teff_object[:, 1] + teff_object[:, 2], 2)
            
            # teff_avrg = [nest_teff, lower, upper]
            teff_avrg.append([np.average(teff_object[:, i], weights=teff_weights) for i in range(3)])

    teff_avrg = np.transpose(np.array(teff_avrg))
    return teff_avrg


def rv_avrg(year, month, dates, frames):
    '''
        Read and calculate the weighted average rvs of one month's observation from MCMC.\n
        year = 20\\
        month = 1\\
        dates = [18, 19, 20]\\
        frames[i][j][k] = i-th date, j-th object, k-th frame\\
        Returns 3*N array: rv_avrg[:, i] = [rv(absolute), lower, upper]
    '''
    
    orders = [32, 33]
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    year = str(year).zfill(2)
    month = str(month).zfill(2)
    rv_avrg = []

    for idx, date in enumerate(dates):
        date = str(date).zfill(2)
            # Line number -1 of rv.
        if int(year) > 19:
            rv_line = 13
        else:
            rv_line = 15
        
        # Special Case
        if int(year)==20 and int(month)==1 and int(date)==19:
            rv_line = 15
        
        for object in frames[idx]:
            rv_object = []
            
            for frame in object:
                
                if int(year) > 18:
                    sci_name = 'nspec' + year + month + date + '_' + str(frame).zfill(4)
                else:
                    sci_name = month_list[int(month) - 1] + date + 's' + str(frame).zfill(4)
                
                #########################################
                ############### Read MCMC ###############
                #########################################
                rv_path = '/home/l3wei/ONC/Data/20' + year + month_list[int(month)-1] + date + '/reduced/mcmc/' + sci_name + '_MultipleOrders_fixedStellarParams_Veiling_addContinuum_KDEMove_' + str(orders) + '_badpix/mcmc_parameters.txt'
                
                if not os.path.exists(rv_path):
                    rv_path = rv_path.replace("_badpix", "")
                
                with open(rv_path) as file:
                    rv_raw = file.readlines()
                    rv_frame = [float(i) for i in rv_raw[rv_line].strip('rv_mcmc () km/s\n').split(', ')]
                    rv_frame[0] = [float(i) for i in rv_raw[rv_line + 1].strip('rv_mcmc () km/s\n').split(', ')][0] # replace rv with absolute rv
                    rv_object.append(rv_frame)
            
            # Calculate weighted average:
            rv_object = np.array(rv_object)
            rv_weights = 1/np.power(rv_object[:, 1] + rv_object[:, 2], 2)
            
            # rv_avrg = [rv(abs), lower, upper], N*3
            rv_avrg.append([np.average(rv_object[:, i], weights=rv_weights) for i in range(3)])
    
    # rv_avrg = [rv(abs); lower; upper], 3*N
    rv_avrg = np.transpose(np.array(rv_avrg))
    return rv_avrg



def logg_avrg(year, month, dates, frames):
    '''
        Read and calculate the weighted average loggs of one month's observation from Multinest.\n
        year = 20\\
        month = 1\\
        dates = [18, 19, 20]\\
        frames[i][j][k] = i-th date, j-th object, k-th frame\\
        Returns 3*i array: logg_avrg[:, i] = [logg, lower, upper]
    '''
    orders = [32, 33]
    output_prefix = 'M_'

    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    year = str(year).zfill(2)
    month = str(month).zfill(2)
    logg_avrg = []

    for idx, date in enumerate(dates):
        date = str(date).zfill(2)
        
        for object in frames[idx]:
            logg_object = []
            
            for frame in object:
                
                if int(year) > 18:
                    sci_name = 'nspec' + year + month + date + '_' + str(frame).zfill(4)
                else:
                    sci_name = month_list[int(month) - 1] + date + 's' + str(frame).zfill(4)
                
                #########################################
                ############# Read Multinest ############
                #########################################
                logg_path = '/home/l3wei/ONC/Data/20' + year + month_list[int(month)-1] + date + '/reduced/multinest/' + sci_name + '_' + str(orders) + '/' + output_prefix + 'Multinest_Params.txt'
                
                with open(logg_path) as file:
                    logg_raw = file.readlines()
                    logg_temp = [float(i) for i in logg_raw[2].strip('logg: \t').split(', ')]
                    logg_object.append(logg_temp) 
            
            # Calculate weighted average:
            logg_object = np.array(logg_object)
            logg_weights = 1/np.power(logg_object[:, 1] + logg_object[:, 2], 2)
            
            # logg_avrg = [nest_logg, lower, upper]
            logg_avrg.append([np.average(logg_object[:, i], weights=logg_weights) for i in range(3)])

    logg_avrg = np.transpose(np.array(logg_avrg))
    return logg_avrg

# if __name__=="__main__":
#     year = 20
#     month = 1
#     dates = [18, 19, 20, 21]
#     HC_IDs = [
#         [217, 229],
#         [228, 224, 135], 
#         [440, 450, 227, 204, 229], 
#         [240, 546, 504, 703, 431, 229]
#     ]
#     frames = [
#         [[27, 28, 29, 30], [33]], 
#         [[31, 32, 33, 34], [37, 38, 39, 40], [41, 42, 43]], 
#         [[32, 33, 34, 35], [36, 37, 38, 39], [42, 43, 44], [46, 47, 48, 49], [50, 51, 52, 53]], 
#         [[33, 34, 35, 36], [39, 40, 41, 42], [43, 44, 45], [47, 48, 50], [53, 54, 55], [58, 59, 60]]
#     ]

#     teff_avrg = teff_avrg(year=year, month=month, dates=dates, frames=frames)
#     rv_avrg = rv_avrg(year=year, month=month, dates=dates, frames=frames)