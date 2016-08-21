import numpy as np

#==========================================================================
#convert sand data from cumulative to rate

def sand_cum_to_rate(full_sand_data):
    sand_diff =  np.ediff1d(full_sand_data[:,1])
    time_data = full_sand_data[:,0][0:len(sand_diff)]
    #full_sand_data[:-1,1] = sand_diff
    #full_sand_data = full_sand_data[:-1]
    
    return np.column_stack((time_data, sand_diff))


#==========================================================================
#bin the wind
def wind_binning(full_wind_data , sand_frequency):
    
    if sand_frequency == '10Hz':
        bin_size = 5
    elif sand_frequency  == '25Hz':
        bin_size = 2

    # points to take off to make data divisible
    del_pts = len(full_wind_data)%bin_size

    # id perfectly divisible then need to change del points for tater
    if del_pts ==0:
        del_pts = -len(full_wind_data)

    # create a matrix of zeros for the binned data
    bin_wind = np.zeros( ( len(full_wind_data)/bin_size , 4) )

    # I will take the five values following ii and add them togther and finally divide
    for ii in range( bin_size ):
        bin_wind += full_wind_data[  ii:-del_pts:bin_size , : ]
    
    return bin_wind/bin_size




#==========================================================================
#find sheer force

#first define a moving average
def moving_average(data , window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / float(window_width)
    return ma_vec

#now onto sheer force
def sheer_force( wind_data , mving_window ):

    vel_primes = np.zeros( (len(wind_data) - mving_window + 1 , 3) )
    for i in range(1,4):
        avg = moving_average(wind_data[:,i] , mving_window)
        uvw_prime = avg -  wind_data[:-mving_window+1 , i]
    
        vel_primes[:,(i-1)] = uvw_prime


    #now we have the primes we need to do a moving avergae of theor products such taht
    up_wp = vel_primes[:,0] *  vel_primes[:,2]
    vp_wp = vel_primes[:,1] *  vel_primes[:,2]

    mvg_avg_up_wp = moving_average(up_wp , mving_window)
    mvg_avg_vp_wp = moving_average(vp_wp , mving_window)

    sheer_force_v = np.sqrt( mvg_avg_up_wp**2 + mvg_avg_vp_wp**2 )
    return sheer_force_v

#==========================================================================
#convert our data into a lsit of vectors objects where the vecotr compoentns
# are made fro the poojt s around the start time of interest

def x_data_time_vec(x_data , lag , start_pt , finish_pt , range_val ):
    #now need to include data from around the lag time
    rng = float( range_val )
    time_indices = np.arange( -int(lag/rng)  , int(lag/rng))
    
    
    len_new_array = finish_pt - start_pt
    x_data_time = np.zeros(   (len_new_array  , len(time_indices)) )
    for time_i in range(len(time_indices)):
        time = time_indices[time_i]
        x_data_time[:,time_i] = x_data[start_pt+time : finish_pt+time]
    
    return x_data_time



