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
from filters import * #high_low_freq_split( data , data_freq , cut_off_freq )
#=========================================================================
#noce tempate I created to plot mutiple plots om same axis

#f , axarr = plt.subplots(3, ncols=2, sharex=True, sharey=False)
#
#for i in range(6):
#    station_a = raw_data( i , '25Hz')
#    station_a_sand = sand_cum_to_rate(station_a[0])
#    time = np.arange(0  , len(station_a_sand))*1./25
#    print i
#
#    axarr.flat[i].plot( time , station_a_sand[:,1] )
#    stations = [1,2,7,8,13,14]
#    axarr.flat[i].set_title('Station %s'%(stations[i]),size = 22)
#    axarr.flat[i].tick_params(axis='y', which = 'major',labelsize=22)
#    axarr.flat[i].tick_params(axis='x', which = 'major',labelsize=22)
#    for label in axarr.flat[i].get_xticklabels()[::2]:
#        label.set_visible(False)
#    for label in axarr.flat[i].get_yticklabels()[::2]:
#        label.set_visible(False)
#
#plt.subplots_adjust(left=0.08, bottom=0.09, right=0.95, top=0.93, wspace=0.12, hspace=None)
#
#left  = 0.125  # the left side of the subplots of the figure
#right = 0.9    # the right side of the subplots of the figure
#bottom = 0.1   # the bottom of the subplots of the figure
#top = 0.9      # the top of the subplots of the figure
#wspace = 0.1   # the amount of width reserved for blank space between subplots
#hspace = 0.5   # the amount of height reserved for white space between subplots
#f.text(0.5, 0.01, 'Time in sec', ha='center' , size = 26)
#f.text(0.01, 0.5, 'Sand Deposit in gram/sec', va='center', rotation='vertical' , size = 26)
#
#
#plt.show()
#
#exit()




##plotting the power-frequency spectrum
#
#
#f , axarr = plt.subplots(3, ncols=2, sharex=True, sharey=False)
#
#for i in range(6):
#    station_a = raw_data( i , '25Hz')
#    station_a_sand = sand_cum_to_rate(station_a[0])[100000:]
#    
#    F = np.fft.fft(station_a_sand[:,1])
#    # get sample frequency in cycles/sec (Hz)
#    freq = np.fft.fftfreq(len(F),1./25)
#    print i
#
#    axarr.flat[i].plot( freq , np.abs(F))
#    stations = [1,2,7,8,13,14]
#    axarr.flat[i].set_title('Station %s'%(stations[i]),size = 22)
#    axarr.flat[i].tick_params(axis='y', which = 'major',labelsize=22)
#    axarr.flat[i].tick_params(axis='x', which = 'major',labelsize=22)
#    for label in axarr.flat[i].get_xticklabels()[::2]:
#        label.set_visible(False)
#    for label in axarr.flat[i].get_yticklabels()[::2]:
#        label.set_visible(False)
#
#plt.subplots_adjust(left=0.08, bottom=0.09, right=0.95, top=0.93, wspace=0.12, hspace=None)
#
#left  = 0.125  # the left side of the subplots of the figure
#right = 0.9    # the right side of the subplots of the figure
#bottom = 0.1   # the bottom of the subplots of the figure
#top = 0.9      # the top of the subplots of the figure
#wspace = 0.1   # the amount of width reserved for blank space between subplots
#hspace = 0.5   # the amount of height reserved for white space between subplots
#f.text(0.5, 0.01, 'Frequency in Hz', ha='center' , size = 26)
#f.text(0.01, 0.5, 'Power', va='center', rotation='vertical' , size = 26)
#
#
#plt.show()
#
#exit()



#==========================================================================
#import data

#select sand data Hz
sand_freq = '25Hz' #either '10Hz' or '25Hz'

#raw data function has the sand as [0] and wind as [1]
station_a = raw_data( 2 , sand_freq)
station_b = raw_data( 3 , sand_freq )

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

sheer_force_a = sheer_force( station_a[1] , 25 )
sheer_force_a = sheer_force_a[1::2]

sheer_force_b = sheer_force( station_b[1] , 25 )
sheer_force_b = sheer_force_b[1::2]

print '-'*10
print 'finished loading the data'
#==========================================================================

#==========================================================================
##plots of the raw data
#
##ydata is the first entry, need to make it an array of all the data
##that needs plotting
##
#Y_plotting = [station_a_sand[:,1]  , station_b_sand[:,1] ]
#multi_plot(Y_plotting)
#
#exit()

#==========================================================================



#==========================================================================
#DEaling with noise using filters


##look at the FT of the data
## take the fourier transform of the data
#F = np.fft.fft(station_a_sand[:,1])
#    
## get sample frequency in cycles/sec (Hz)
#f = np.fft.fftfreq(len(F),1./25)
#plt.plot( f , np.abs(F) )
#plt.show()
#exit()


#cleaning the data with the filter desired
cleaned_sand_a , noise_sand_a = high_low_freq_split(station_a_sand[:,1][100000:] , 25 , 10)
cleaned_sand_b , noise_sand_b = high_low_freq_split(station_b_sand[:,1][100000:] , 25 , 10)

print '-'*10
print 'finished cleaning the data'
print '-'*10

#Y_plotting = [station_a_sand[:,1][100000:], cleaned_sand_a , station_b_sand[:,1][100000:], cleaned_sand_b   ]
#multi_plot(Y_plotting)
#exit()


#cleaned_sand_a_2 , noise_sand_a = high_low_freq_split(station_a_sand[:,1][100000:] , 25 , 2)
#time = np.arange(0,len(cleaned_sand_a))*(1./25)
#
#f , axarr = plt.subplots(3, ncols=1, sharex=True, sharey=False)
#
#axarr.flat[0].plot( time , station_a_sand[:,1][100000:] )
#axarr.flat[0].tick_params(axis='y', which = 'major',labelsize=22)
#axarr.flat[0].tick_params(axis='x', which = 'major',labelsize=22)
#axarr.flat[0].set_title('Unfiltered Data',size = 22)
#
#for label in axarr.flat[0].get_xticklabels()[::2]:
#    label.set_visible(False)
#for label in axarr.flat[0].get_yticklabels()[::2]:
#    label.set_visible(False)
#
#
#
#axarr.flat[1].plot( time , cleaned_sand_a  )
#axarr.flat[1].tick_params(axis='y', which = 'major',labelsize=22)
#axarr.flat[1].tick_params(axis='x', which = 'major',labelsize=22)
#axarr.flat[1].set_title('Filtered Data at 10Hz Cutoff',size = 22)
#for label in axarr.flat[1].get_xticklabels()[::2]:
#    label.set_visible(False)
#for label in axarr.flat[1].get_yticklabels()[::2]:
#    label.set_visible(False)
#
#
#
#axarr.flat[2].plot( time , cleaned_sand_a_2  )
#axarr.flat[2].tick_params(axis='y', which = 'major',labelsize=22)
#axarr.flat[2].tick_params(axis='x', which = 'major',labelsize=22)
#axarr.flat[2].set_title('Filtered Data at 2Hz Cutoff',size = 22)
#for label in axarr.flat[2].get_xticklabels()[::2]:
#    label.set_visible(False)
#for label in axarr.flat[2].get_yticklabels()[::2]:
#    label.set_visible(False)
#
#
#
#plt.subplots_adjust(left=0.08, bottom=0.09, right=0.95, top=0.93, wspace=0.12, hspace=None)
#
#left  = 0.125  # the left side of the subplots of the figure
#right = 0.9    # the right side of the subplots of the figure
#bottom = 0.1   # the bottom of the subplots of the figure
#top = 0.9      # the top of the subplots of the figure
#wspace = 0.1   # the amount of width reserved for blank space between subplots
#hspace = 0.5   # the amount of height reserved for white space between subplots
#
#f.text(0.5, 0.01, 'Time in sec', ha='center' , size = 26)
#f.text(0.01, 0.5, 'Sand Deposit in gram/sec', va='center', rotation='vertical' , size = 26)
#
#
#plt.show()
#
#exit()




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


#==========================================================================
#calculate MI

#the first value is 'y' for nomralized one.
#this function will force equal length, the wind should be the second input
#the last value si the total lag needed

#total_lag = 300
#bins = 1000
##MI_vec =  MI_lag(bins ,  station_a_sand[:,1][100000:] , windspeed_station_a[100000:] , total_lag)
##MI_vec =  MI_lag(bins ,  station_a_sand[:,1][100000:] , sheer_force_a[1::2][100000:] , total_lag)
#MI_vec =  MI_lag(bins ,  cleaned_sand_a[100000:] ,  cleaned_sand_b[100000:] , total_lag)
#
#
#plt.plot(MI_vec)
#plt.show()
#
#exit()



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
    
    time_indices = np.arange( -int(lag/2.)  , int(lag/2.))
    
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
        val = np.exp( rate_v[i]*one_data_pt  )
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


from numpy.linalg import inv
#define a function to find uncertainty from Bayes
def S(Phi , alpha,beta):
    multiply = np.dot( Phi.T  , Phi )
    alpha_trm = alpha*np.identity(len(Phi.T) )
    return inv( alpha_trm + beta*multiply )

def uncertainty( x_new , Phi , alpha , beta  , num_of_params , rate_min , rate_max):
    SN = S(Phi , alpha,beta)
    phi_trm = fit_func(x_new ,num_of_params , rate_min , rate_max)
    multi_one = np.dot( SN , phi_trm )
    multi_two = np.dot( phi_trm.T  , multi_one)
    return 1./beta + multi_two



lag = 22
num_parameters = 5
l_rate = 0.0
h_rate = l_rate+0.5


#Learning the model

data = x_data_time_vec(windspeed_station_a , lag , 101000 , 300000)
#data = x_data_time_vec(sheer_force_a , lag , 101000 , 300000)



phi_X = Phi(data , num_parameters ,l_rate , h_rate)


from sklearn import linear_model

clf = linear_model.BayesianRidge()
#clf.fit(phi_X, station_a_sand[:,1][100000+lag:300000+lag])
#try with cleaned data
clf.fit(phi_X, cleaned_sand_a[1000+lag:200000+lag])



#making and plotting the predictions

data_2 = x_data_time_vec(windspeed_station_b , lag , 101000 , 300000)
#data_2 = x_data_time_vec(sheer_force_b , lag , 100100 , 300000)

phi_x2  = Phi(data_2 , num_parameters , l_rate , h_rate)

predicted_sand = clf.predict( phi_x2 )


print '-'*10
print 'finished making prediction'
print '-'*10


##Y_plotting = [predicted_sand , station_b_sand[:,1][100000+lag:200000+lag]]
#Y_plotting = [ predicted_sand  , cleaned_sand_b[1000+lag:200000+lag]   ]
#multi_plot(Y_plotting)


unc_vec = []
for i in range(len(data_2)):
    unc = uncertainty( data_2[i] , phi_x2 , clf.lambda_ , clf.alpha_  , num_parameters , l_rate , h_rate)
    unc_vec.append(unc)
unc_vec = np.array(unc_vec)


plt.plot( predicted_sand )
plt.plot( cleaned_sand_b[1000+lag:200000+lag] )

plt.fill_between( predicted_sand+ np.sqrt(unc_vec) , predicted_sand - np.sqrt(unc_vec), facecolor='#FFB6C1', alpha=1.0, edgecolor='none')


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




