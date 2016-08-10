import numpy as np

#allows me to pull matlab fles
import scipy.io


#creates a function that is used to call the data in a numpy format.
#Note the following:
#a) Stations 1 , 2 , 7 , 8 , 13 , 14  are equivilent to a choice of station number = 0 , 1 , 2,  ,3 , 4 , 5 respectively
#b) sand type can be the string '10Hz' or '25Hz'

def raw_data( station_number , sand_type ):

    #==========================================================================
    #STRINGS FOR CHOSING THE CORRECT FILES


    #make a string of the name of the differnt files and folders:
    files_names = ['data_files/station1/station1_' , 'data_files/station2/station2_' , 'data_files/station7/station7_' , 'data_files/station8/station8_' , 'data_files/station13/station13_' , 'data_files/station14/station14_']

    #a string of the different file types within each file
    file_type = ['sandcumul10Hz' , 'sandcumul25Hz' , 'wind50Hzuvw']

    #==========================================================================

    #==========================================================================
    #load the sand and wind data from the stations of our choice
    
    if sand_type == '10Hz':
        sand = 0
    elif sand_type  == '25Hz':
        sand = 1

        
    sand_file_name = "".join(( '../' ,files_names[station_number], file_type[sand]))
    wind_file_name = "".join(('../' ,files_names[station_number], file_type[2]))
        
    sand_data = np.array( scipy.io.loadmat(sand_file_name)[file_type[sand]] )
    wind_data = np.array( scipy.io.loadmat(wind_file_name)[file_type[2]] )
        
    return [sand_data , wind_data]


    #==========================================================================