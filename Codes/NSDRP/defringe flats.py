##################################################
# The example to demonstrate the defringe function 
##################################################
## Import the smart package
import smart
import os

## Set up the data path
data_folder_path = '/home/l3wei/ONC/Data/2022jan20/specs/'

## Run the defringe function
smart.defringeflatAll(data_folder_path, diagnostic=False, start_col=10, end_col=2048-50)
