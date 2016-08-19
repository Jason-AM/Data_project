import numpy as np
from scipy.signal import wiener , butter, lfilter, freqz


#=========================================================================
#keeps data
def manual_filter_1( data , data_freq , cut_off_freq ):
    
    #get time step from frequency
    dt = 1./data_freq
    
    # take the fourier transform of the data
    F = np.fft.fft(data)
    
    # get sample frequency in cycles/sec (Hz)
    f = np.fft.fftfreq(len(F),dt)

    #positions where we keep the data
    pos_of_keep = np.where( np.logical_and( f.real < cut_off_freq  , f.real > -cut_off_freq ) )

    #get the data we are keeping
    F_keep = F[pos_of_keep]

    #do the inverse fft to get the data back where we pad with zeros
    data_filtered = np.fft.ifft(F_keep , len(F)).real

    return data_filtered


#=========================================================================
#throws away data
def manual_filter_2( data , data_freq , cut_off_freq ):
    
    #get time step from frequency
    dt = 1./data_freq
    
    # take the fourier transform of the data
    F = np.fft.fft(data)
    
    # get sample frequency in cycles/sec (Hz)
    f = np.fft.fftfreq(len(F),dt)
    
    #positions where we throw away the data
    pos_of_throw = np.where( np.logical_or( f.real > cut_off_freq  , f.real < -cut_off_freq ) )
    
    #set the thrown positions to 0
    F[pos_of_throw] = 0
    
    #do the inverse fft to get the data back
    data_filtered = np.fft.ifft(F).real

    return data_filtered



#=========================================================================
#splits data into high and low frewuency parts
def high_low_freq_split( data , data_freq , cut_off_freq ):
    
    #get time step from frequency
    dt = 1./data_freq
    
    # take the fourier transform of the data
    F = np.fft.fft(data)
    
    # get sample frequency in cycles/sec (Hz)
    f = np.fft.fftfreq(len(F),dt)
    
    #make 2 copies of F to make into what we need
    F_clean = np.copy( F )
    F_noise = np.copy( F )
    
    #positions where we throw away the data
    pos_of_throw = np.where( np.logical_or( f.real > cut_off_freq  , f.real < -cut_off_freq ) )
    
    #positions where we keep the data
    pos_of_keep = np.where( np.logical_and( f.real < cut_off_freq  , f.real > -cut_off_freq ) )
    
    #mkae the cleaned up data frequencies then trasform to get cleaned data
    F_clean[pos_of_throw] = 0
    data_cleaned = np.fft.ifft(F_clean).real
    
    #mkae the noise data frequencies then trasform to get noise data
    F_noise[pos_of_keep] = 0
    data_noise= np.fft.ifft(F_noise).real
    

    
    return data_cleaned , data_noise

#=========================================================================


#=========================================================================
#Wienner filter
def wiener_filter( data , data_freq , cut_off_freq ):
    
    return  wiener(data, mysize=None, noise=None)


#=========================================================================
#butter online filter

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_fiter( data , data_freq , cut_off_freq , order ):

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cut_off_freq, data_freq , order)

    T = len(data)/float(data_freq )         # seconds
    n = int(T * data_freq ) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)


    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cut_off_freq, data_freq, order)

    return y
#=========================================================================
