import numpy as np
import csv, re
import os
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

def model_func(x, a, b, c):
    return a * np.exp(b*x) + c


def convert_to_csv():
    with open('pm_hillenbrand.txt', 'r') as file:
        lines = file.readlines()
    
    result = []
    for line in lines:
        result_line = [_.strip() for _ in re.split('\t|  ', line) if _.strip()]
        result_line = ['' if _=='-1000' else _ for _ in result_line]
        result.append(result_line)
    
    with open('pm_hillenbrand.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(result)


############### Customize Targets ###############

year = '2023'
month = 'jan'

targets = [
    13, 9, 6, 7, 2, 10, 18, 22, 12, 27, 29, 47, 59, 78, 58, 79, 102
]

# targets = list(range(1, 103))

save_path = year + month

if not os.path.exists(save_path):
    os.makedirs(save_path)

# convert_to_csv()
table = pd.read_csv('pm_hillenbrand.csv', dtype=str)

target_kmag = [table['K'][list(table['#']).index(str(target))] for target in targets]
target_hcid = [table['H#'][list(table['#']).index(str(target))] for target in targets]

target_kmag = [float(_) for _ in target_kmag]


model_kmag = []
model_itime = []
with open('Kmag-itime.csv', 'r') as file:
    raw = csv.reader(file)
    next(raw, None)
    for line in raw:
        model_kmag.append(float(line[0]))
        model_itime.append(int(line[1]))

model_kmag = np.array(model_kmag)
model_itime = np.array(model_itime)
plt.scatter(model_kmag, model_itime)

# popt: parameter optimal
popt, pcov = curve_fit(model_func, model_kmag, model_itime)

xdata = np.linspace(min(model_kmag), max(model_kmag), 50)
plt.plot(xdata, model_func(xdata, *popt))
plt.xlabel('kmag')
plt.ylabel('itime')
plt.show()


itimes = np.unique(model_itime).astype(int)
target_itime = []

for kmag in target_kmag:
    distance = np.abs(model_func(kmag, *popt) - itimes)
    target_itime.append(itimes[np.argmin(distance)])


print('\t'.join(['#', 'H#', 'K', 'itime']))
for a, b, c, d in zip(targets, target_hcid, target_kmag, target_itime):
    print('\t'.join([str(_) for _ in [a, b, c, d]]))


with open(save_path + '/' + year + ' ' + month + ' observing plan.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['#', 'H#', 'K', 'itime'])
    
    for a, b, c, d in zip(targets, target_hcid, target_kmag, target_itime):
        writer.writerow([a, b, c, d])