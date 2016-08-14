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
from filters import * #import manual_filter_1 , manual_filter_2 ,  wiener_filter , butter_fiter


#==========================================================================
#import data

##select sand data Hz
#sand_freq = '25Hz' #either '10Hz' or '25Hz'
#
##raw data function has the sand as [0] and wind as [1]
#station_a = raw_data( 0 , sand_freq)
#station_b = raw_data( 1 , sand_freq )
#
##The sand data
#station_a_sand = sand_cum_to_rate(station_a[0])
#station_b_sand = sand_cum_to_rate(station_b[0])
#
##becareful, the wind data is either one too big or one too small for equality
#station_a_bin_wind = wind_binning( station_a[1]  , sand_freq)
#station_b_bin_wind = wind_binning( station_b[1]  , sand_freq)
#
##get the wind speed and then cut th data by 2 for frequency mtching
#windspeed_station_a = np.sqrt( station_a[1][:,1]**2  + station_a[1][:,2]**2  + station_a[1][:,3]**2 )
#windspeed_station_a = windspeed_station_a[1::2]
#
#windspeed_station_b = np.sqrt( station_b[1][:,1]**2  + station_b[1][:,2]**2  + station_b[1][:,3]**2 )
#windspeed_station_b = windspeed_station_b[1::2]
#
#
#
##find sheer stress
##we will use a moving average over 20 frames (20 from corelation at 25 HZ, this is 50hz so 40)
#
#sheer_force_a = sheer_force( station_a[1] , 40 )
#sheer_force_a = sheer_force_a[1::2]
#
#sheer_force_b = sheer_force( station_b[1] , 40 )
#sheer_force_b = sheer_force_b[1::2]





#raw data function has the sand as [0] and wind as [1]


#number of plots required:
    
stations = ['1' , '2' , '7' , '8' , '13' , '14']
f , axarr = plt.subplots(3, ncols=2, sharex = True, sharey = False)
    
for i in np.arange(0 , 6 ):
    
    station_a = raw_data( i , '25Hz')
    station_a_sand = sand_cum_to_rate(station_a[0])
    
    time = np.arange(len(station_a_sand))*(1./25)

    y_plotting = station_a_sand[:,1]
    x_plotting = time
    
    axarr.flat[i].plot(x_plotting , y_plotting)
    axarr.flat[i].set_title('Station %s'%(stations[i]))

plt.show()


exit()
##new =  manual_filter_2(station_a_sand [:,1], 25 , 11.2)
#new = butter_fiter(station_a_sand [:,1], 25 , 11 , 6)
#Y = [new , station_a_sand [:,1]]
#
#multi_plot(Y )
#exit()

#==========================================================================
#The fft


#
s_measured = station_a_sand[:,1]
dt = 1./25
f_signal  = 1   # signal frequency  in Hz

# take the fourier transform of the data
F_2 = np.fft.fft(s_measured)


# calculate the frequencies for each FFT sample
f = np.fft.fftfreq(len(F_2),dt)  # get sample frequency in cycles/sec (Hz)

#cut_off = 2
#
#pos_of_keep = np.where( np.logical_and( f.real < cut_off  , f.real > -cut_off ) )
#pos_of_throw = np.where( np.logical_or( f.real > cut_off  , f.real < -cut_off ) )
#
#
##F_keep = F_2[pos_of_keep]
##F_2[pos_of_throw] = 0
##f_keep = f[pos_of_keep]
#
#F_noise = F_2[pos_of_throw]
#
#
#
##s_guessed = np.fft.ifft(F_keep , len(F_2))
##s_guessed = np.fft.ifft(F_2 , len(F_2))
#s_guessed = np.fft.ifft(F_noise)
#
#Y = [s_guessed.real , s_measured]
###X = [f_keep , f_keep]
#multi_plot(Y  )
#exit()
###
#
#
### filter the Fourier transform
##def filter_rule(x,freq):
##    band = 0.05
##    if abs(freq)>f_signal+band or abs(freq)<f_signal-band:
##        return 0
##    else:
##        return x
##
##F_filtered = array([filter_rule(x,freq) for x,freq in zip(F,f)])
##
### reconstruct the filtered signal
##s_filtered = ifft(F_filtered)
##
#### get error
###err = [abs(s1-s2) for s1,s2 in zip(s,s_filtered) ]
#
##
from scipy.signal import wiener
s_measured = station_a_sand[:,1]
#Y_plotting = [wiener(s_measured, mysize=None, noise=100), s_measured ]
##X_plotting = [f , f]
#multi_plot(Y_plotting )



#==========================================================================
#TRandom online



from scipy.signal import butter, lfilter, freqz



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    print nyq
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



# Filter requirements.
order = 6
fs = 25 #30       # sample rate, Hz
cutoff = 2 #3.677  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
#b, a = butter_lowpass(cutoff, fs, order)
b, a = butter_highpass(cutoff, fs, order)

data = station_a_sand[:,1]
T = len(data)/float(fs)         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)


# Filter the data, and plot both the original and filtered signals.
y_low = butter_lowpass_filter(data, cutoff, fs, order)
y_high = butter_highpass_filter(data, cutoff, fs, order)
#y_high = np.abs(y_high)
y_high[y_high < 0] = 0

Y_plotting = [data, y_low + y_high ,y_low , y_high  ]#, s_guessed.real  , wiener(s_measured, mysize=None, noise=None) ]
#X_plotting = [f , f]
multi_plot(Y_plotting )


#plt.plot( f , np.fft.fft(y_high) )
#plt.show()

