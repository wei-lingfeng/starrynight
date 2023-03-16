import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import interpolate

MT_Model = []
masses = np.concatenate((np.arange(0.5, 2.1, 0.1), np.array([2.2, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0])))
masses = ['%.1f' %i for i in masses]
# masses = ['0.5','0.6','0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.2', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0']

for m in masses:
    TA = []
    with open('/home/l3wei/ONC/Data/siess_for_lingfeng/s' + m + '_02m1.5.hrd', 'r') as file:
        lines = file.readlines()
        for line in lines:
            temp = [float(i) for i in line.split()]
            if temp[10] > 1e7:
                break
            temp = [temp[10], temp[6]] # [Age, Teff]
            TA.append(temp)

    TA = np.array(np.transpose(TA))
    TA_Func = interpolate.interp1d(TA[0, :], TA[1, :])
    MT_Model.append([float(TA_Func(2e6)), float(m)])

MT_Model = np.array(np.transpose(MT_Model))
plt.plot(MT_Model[0,:], MT_Model[1,:])

# with open('/home/l3wei/ONC/Models/Seiss_M-T_Model.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(np.transpose(MT_Model))