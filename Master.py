import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mutual_info_score , normalized_mutual_info_score
from sys import path

#==========================================================================
#importing my own functions


path.append('./functions/')

from Raw_data import raw_data
from Refrom_data import wind_binning , sand_cum_to_rate , sheer_force
from MI_lag import MI_lag
from multiple_plots import multi_plot
from filters import * #manual_filter_1 , manual_filter_2 ,  wiener_filter , butter_fiter


#==========================================================================
#import data

#select sand data Hz
sand_freq = '25Hz' #either '10Hz' or '25Hz'

#raw data function has the sand as [0] and wind as [1]
station_a = raw_data( 2 , sand_freq)
station_b = raw_data( 0 , sand_freq )

#The sand data
station_a_sand = sand_cum_to_rate(station_a[0])
station_b_sand = sand_cum_to_rate(station_b[0])

#becareful, the wind data is either one too big or one too small for equality
station_a_bin_wind = wind_binning( station_a[1]  , sand_freq)
station_b_bin_wind = wind_binning( station_b[1]  , sand_freq)

#get the wind speed and then cut th data by 2 for frequency mtching
windspeed_station_a = np.sqrt( station_a[1][:,1]**2  + station_a[1][:,2]**2  + station_a[1][:,3]**2 )
windspeed_station_a = windspeed_station_a[1::2]

windspeed_station_b = np.sqrt( station_b[1][:,1]**2  + station_b[1][:,2]**2  + station_b[1][:,3]**2 )
windspeed_station_b = windspeed_station_b[1::2]



#find sheer stress
#we will use a moving average over 20 frames (20 from corelation at 25 HZ, this is 50hz so 40)

sheer_force_a = sheer_force( station_a[1] , 250 )
sheer_force_a = sheer_force_a[1::2]

sheer_force_b = sheer_force( station_b[1] , 250 )
sheer_force_b = sheer_force_b[1::2]



#==========================================================================


#possies = np.where(np.logical_and(windspeed_station_a[:-21]>=4.999, windspeed_station_a[:-21]<=5.001))
#plt.plot( windspeed_station_a[:-21][possies] , station_a_sand[:,1][21:][possies] , 'o' )
#
#possies = np.where(np.logical_and(sheer_force_a[:-21]>=0.999, sheer_force_a[:-21]<=1.001))
#plt.plot( sheer_force_a[:-21][possies] , station_a_sand[:,1][21:][possies] , 'o' )
#print possies
#
#
#plt.show()
##exit()

#==========================================================================
#plots of the raw data

#ydata is the first entry, need to make it an array of all the data
#that needs plotting

#Y_plotting = [sheer_force_a , station_a_sand[:,1] , windspeed_station_a ]
#multi_plot(Y_plotting)



#==========================================================================


#==========================================================================
#calculate MI

##the first value is 'y' for nomralized one.
##this function will force equal length, the wind should be the second input
##the last value si the total lag needed
#
#total_lag = 100
#bins = 1000
##MI_vec =  MI_lag(bins ,  station_a_sand[:,1][100000:] , windspeed_station_a[100000:] , total_lag)
#MI_vec =  MI_lag(bins ,  station_a_sand[:,1][100000:] , sheer_force_a[1::2][100000:] , total_lag)
#
#plt.plot(MI_vec)
#plt.show()
#
#exit()

#==========================================================================
#DEaling with noise using filters


##used to plot the fft of the data, used to see where cutoffs should be
#fft_a = np.fft.fft( station_a_sand[:,1] )
#f_a = np.fft.fftfreq(len(fft_a),1./25)
#
#
#fft_b = np.fft.fft( station_b_sand[:,1] )
#f_b = np.fft.fftfreq(len(fft_b),1./25)
#
#Y_plotting = [fft_a.real , fft_a.imag , fft_b.real , fft_b.imag ]
#X = [f_a,f_a , f_b , f_b]
#multi_plot(Y_plotting , X)
#exit()

#cleaning the data with the filter desired
cleaned_sand_a = butter_fiter(station_a_sand[:,1] , 25 , 2 , 7)
cleaned_sand_b = butter_fiter(station_b_sand[:,1] , 25 , 2 , 7)


Y_plotting = [sheer_force_a , cleaned_sand_a , windspeed_station_a ]
multi_plot(Y_plotting)
exit()



#Remove the noise which are define by the largest and smallest events in the
# first 20000 positions

#maxi = np.max( np.abs(station_a_sand[:,1][:20000]) )
#
#noise_pos = np.where( np.abs(station_a_sand[:,1]) < maxi )
#cleaned_sand_a = station_a_sand[:,1]
#cleaned_sand_a[noise_pos] = 0
#
#
#maxi = np.max( np.abs(station_b_sand[:,1][:20000]) )
#
#noise_pos = np.where( np.abs(station_b_sand[:,1]) < maxi )
#cleaned_sand_b = station_b_sand[:,1]
#cleaned_sand_b[noise_pos] = 0
#
#
#
##now I would like to find the places where the ratios exceed 10 and kill that data
#bucket_range = 30
#lag = 20
#for i in np.arange(lag+bucket_range,len( cleaned_sand_a )):
#    rng = cleaned_sand_b[i-lag-bucket_range:i-lag+bucket_range]
#    max_pos = np.argmax( np.abs(rng) )
#    max_val = float(rng[max_pos])
#    if max_val != 0:
#        ratio = cleaned_sand_a[i]/max_val
#        if ratio > 10 or ratio < -5:
#            cleaned_sand_a[i] = 0

#bucket_range = 30
#lag = 20
#for i in np.arange(len( cleaned_sand_b ) - lag - bucket_range):
#    rng = cleaned_sand_a[i+lag-bucket_range:i+lag+bucket_range]
#    max_pos = np.argmax( np.abs(rng) )
#    max_val = float(rng[max_pos])
#    if max_val != 0:
#        ratio = cleaned_sand_b[i]/max_val
#        if ratio > 10 or ratio < -5:
#            cleaned_sand_b[i] = 0

#==========================================================================

#==========================================================================
#try Bayesian fitting - basic first attempt Gaussian

#
#
#def phi(x_data , M): #note xdata is a vector
#
#    #range of means used
##    xdata_change = np.max(x_data) - np.min(x_data)
##    size_of_jumps = xdata_change/float(M)
##    mu_i = size_of_jumps/2.
##    s = np.sqrt(size_of_jumps)
#
#    phi_mat = np.zeros((len(x_data) , M))
#    #make the step zise of mu 10% of value
#    mu_i_change = 0.05*x_data
#    mu_i = x_data - (M/2.)*mu_i_change
#    s = np.sqrt(mu_i_change)
#    for i in range(M):
#        #define the function used
#        
#        val = np.exp( -(x_data - mu_i)**2/(2*s) )
#        mu_i += mu_i_change
#        #val = x_data**i
#        phi_mat[:,i] = val
#    return phi_mat
#
#
#X = phi(windspeed_station_a[100000:200000] , 49)
#from sklearn import linear_model
#clf = linear_model.BayesianRidge()
#clf.fit(X, station_a_sand[:,1][100020:200020])
#print clf.coef_
#print clf.alpha_
#print clf.lambda_
#
#
#
#f , axarr = plt.subplots(2, ncols=1, sharex=True, sharey=False)
##
#ax_s_1 = axarr.flat[0]
#ax_s_1.plot(clf.predict( phi(windspeed_station_b[100000:200000] , 49) ))
##
#ax_s_2 = axarr.flat[1]
#ax_s_2.plot( station_b_sand[:,1][100020:200020] )
#
#plt.show()
#exit()

#=========================================================================

#
#try Bayesian fitting second, hectic attempt number 2, efficient


#first make data into a vector from the time lag

def x_data_time_vec(x_data , lag , start_pt , finish_pt ):
    #now need to include data from around the lag time
    
    time_indices = np.arange( -int(lag/4.)  , int(lag/4.))
    
    len_new_array = finish_pt - start_pt
    x_data_time = np.zeros(   (len_new_array  , len(time_indices)) )
    for time_i in range(len(time_indices)):
        time = time_indices[time_i]
        x_data_time[:,time_i] = x_data[start_pt+time : finish_pt+time]

    return x_data_time


#define an exponential function to use in fitting
#should create a row of Phi
def fit_func(one_data_pt , num_of_params , rate_min , rate_max):
    rate_v = np.arange( rate_min  , rate_max , (rate_max - rate_min)/float(num_of_params) )
    phi_row = []
    for i in range(num_of_params):
        val = np.exp( -rate_v[i]*one_data_pt  )
        phi_row.append(val)
    #we need to flatten so that each time element of the input is
    #given a differnt weight.
    return np.array( phi_row ).flatten()

#now define phi itself
def Phi(input , num_of_params , rate_min , rate_max):
    Phi = []
    for i in range( len(input) ):
        one_data_pt = input[i]
        row = fit_func(one_data_pt , num_of_params , rate_min , rate_max)
        Phi.append(row)

    return np.array( Phi )




lag = 21
num_parameters = 10
l_rate = 0.1
h_rate = l_rate+1.5


#Learning the model

data = x_data_time_vec(windspeed_station_a , lag , 100000 , 300000)
#data = x_data_time_vec(sheer_force_a , lag , 100000 , 300000)

phi_X = Phi(data , num_parameters ,l_rate , h_rate)


from sklearn import linear_model

clf = linear_model.BayesianRidge()
#clf.fit(phi_X, station_a_sand[:,1][100000+lag:300000+lag])
#try with cleaned data
clf.fit(phi_X, cleaned_sand_a[100000+lag:300000+lag])



#making and plotting the predictions

data_2 = x_data_time_vec(windspeed_station_b , lag , 100000 , 200000)
#data_2 = x_data_time_vec(sheer_force_b , lag , 100000 , 200000)

phi_x2  = Phi(data_2 , num_parameters , l_rate , h_rate)

#Y_plotting = [clf.predict( phi_x2 ) , station_b_sand[:,1][100000+lag:200000+lag]]
Y_plotting = [clf.predict( phi_x2 ) , cleaned_sand_b[100000+lag:200000+lag]   ]
multi_plot(Y_plotting)

exit()





#=========================================================================

#
#try Bayesian fitting second, hectic attempt


#first make data into a vector from the time lag
#
#
#def phi(x_data_time , num_funcs ): #note xdata is a vector
#    
#    M = num_funcs
#    X = []
#    for data_pt in range(len(x_data_time)):
#        phi_row = []
#        rate = np.arange( 0.8 , 1.2 , 0.4/M )
#        for coeff in range(M):
#            
#            vec_data = x_data_time[data_pt]
#            
##            mu_i_change = 0.05
##            mu_i = 0 - (M/2.)*mu_i_change
##            s = 0.01
##
##            val = np.exp( -( vec_data - mu_i)**2/(2*s) )
##            mu_i += mu_i_change
#            #val = vec_data**coeff
#
#            val = np.exp( -rate[coeff]*vec_data )
#            
#            phi_row = np.append(phi_row , val)
#        
#        X.append(phi_row)
#
#    return np.array(X)
#
#
#
#
#lag = 25
#num_parameters = 10
#
#
#
##Learning the model
#
##data = x_data_time_vec(windspeed_station_a , lag , 100000 , 300000)
#data = x_data_time_vec(sheer_force_a , lag , 100000 , 300000)
#
#phi_X = phi(data , num_parameters)
#
#from sklearn import linear_model
#clf = linear_model.BayesianRidge()
##clf.fit(phi_X, station_a_sand[:,1][100000+lag:300000+lag])
##try with cleaned data
#clf.fit(phi_X, cleaned_sand_a[100000+lag:300000+lag])
#print clf.coef_
#print clf.alpha_
#print clf.lambda_
#print clf.intercept_
#
#
#
##making and plotting the predictions
#
##data_2 = x_data_time_vec(windspeed_station_b , lag , 100000 , 200000)
#data_2 = x_data_time_vec(sheer_force_b , lag , 100000 , 200000)
#phi_x2 = phi(data_2 , num_parameters)
#
#
##Y_plotting = [clf.predict( phi_x2 ) , station_b_sand[:,1][100000+lag:200000+lag]]
#Y_plotting = [clf.predict( phi_x2 ) , cleaned_sand_b[100000+lag:200000+lag]   ]
#multi_plot(Y_plotting)
#
#exit()


#==========================================================================





#==========================================================================
#try Bayesian fitting - basic third attempt - exponential



#def phi(x_data , M): #note xdata is a vector
#
#
#    phi_mat = np.zeros((len(x_data) , M))
#    #make the step zise of mu 10% of value
#    rate = np.arange( 0.5 , 1.5 , 1./M )
#    for i in range(M):
#        #define the function used
#        val = np.exp( -rate[i]*x_data )
#        phi_mat[:,i] = val
#    return phi_mat
#
#
#
#X = phi(windspeed_station_a[100000:200000], 30)
#from sklearn import linear_model
#clf = linear_model.BayesianRidge()
#clf.fit(X, station_a_sand[:,1][100020:200020])
#print clf.coef_
#print clf.alpha_
#print clf.lambda_
#
#
#
#f , axarr = plt.subplots(2, ncols=1, sharex=True, sharey=False)
##
#ax_s_1 = axarr.flat[0]
#ax_s_1.plot(clf.predict( phi(windspeed_station_b[100000:200000] , 30) ))
##
#ax_s_2 = axarr.flat[1]
#ax_s_2.plot( station_b_sand[:,1][100020:200020] )
#
#plt.show()
#exit()
#==========================================================================




