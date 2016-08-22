import numpy as np
import os


#============================================================
#save file to path
def save_data(data , file_name):
    #folder to save in
    folder = '../Produced_data/'
    file_name = folder + file_name
    
    if os.path.exists(file_name+'.npy') == False:
        np.save(file_name , data)
    else:
        print 'whoops, file already exists'



#============================================================
#open file from path
def open_file( file_name ):
    folder = '../Produced_data/'
    file_name = folder + file_name + '.npy'

    return np.load(file_name)

#============================================================