from scipy import signal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy
from ppgtools import sigplot

def resample(inp, fs, tfs, aa = False):
        '''
        Resample the data to a new sampling rate using Fourier method.
        Recommend tfs being an integer factor of fs.

        Parameters
        ----------
        inp : array_like
            Signal to resample.
        fs : int
            Original sample rate.
        tfs : int
            Target sample rate.
        aa : bool, optional
            Set to true if an anti-aliasing filter should be used before resampling. The default is False.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = copy.deepcopy(inp)
        
        if(aa):
            output = filter_signal(output, 'lowpass', [tfs/2], fs)
            
        output = signal.resample(output, int(tfs / fs) * len(output))
        
        return output

def downsample(inp, fs, tfs, aa = False):
    '''
    Downsamples the data to a new sampling rate by decimation.
    Recomend the target sampling rate to be an integer factor of the original.

    Parameters
    ----------
    inp : array_like
            Signal to resample.
    fs : int
        Original sample rate.
    tfs : int
        Target sample rate in Hz.
    aa : bool, optional
        Set to true if an anti-aliasing filter should be used before downsampling. The default is False.

    Returns
    -------
    output : array_like
            The data after the change.

    '''
    inp_filt = inp
    
    #Filter the data to half the target sampling rate
    if(aa):
        inp_filt = filter_signal(inp_filt, 'lowpass', [tfs/2], fs)
    
    modulus = int(fs / tfs)
    output = []
    for i in range (0, len(inp)):
        if(i % modulus == 0):
            output.append(inp[i])
    
    output.append(inp_filt[len(inp_filt)-1])
    output = np.asarray(output)
            
    return output
        
    
def upsample(inp, fs, num_padding):
    '''
    Helper function that will upsample the data to a higher sampling rate.

    Parameters
    ----------
    inp : array_like
            Signal to resample.
    fs : int
        Original sample rate.
    num_padding : int
        How many zeros to put between each data point.

    Returns
    -------
    output : array_like
The data after the change.

    '''

    output = []
    
    for i in range (0, len(inp)):
        output.append(inp[i])
        for j in range(0, num_padding):
            output.append(0)
    
    output = np.asarray(output)
    output = filter_signal(inp, fs, 'lowpass', [(fs/2)-0.1])
        
    return output        
 

def remove_mean(inp):
    '''
    Removes the mean from the data i.e., make the DC offset 0.
    
    Parameters
    ----------
    inp : array_like
            Signal to resample.
    
    Returns
    -------
    output : array_like
            The data after the change.
    
    '''
    out = np.asarray(inp)
    mean = np.mean(inp)
    
    return out - mean
 
 
def filter_signal(inp, fs, filter_pass, fc, order = 4, filter_design = "butter", zero_phase = True):  
    '''
    Filters the signal

    Parameters
    ----------
    inp : array_like
        Signal to resample.
    fs : int
        Original sample rate.
    filter_pass : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        Filter type
    fc : array_like
        Critical freqencies.
    order : int, optional
        The order of the filter. The default is 4.
    filter_design : {'butter', 'bessel', 'chebyii', 'fir'}, optional
        How the filter coefficients are calculated. The default is "butter".
    zero_phase : bool, optional
        If true, uses filtfilt. Otherwise, uses lfilter. The default is True.
        
    Raises
    ------
    ValueError
        If a filter_design is not recognized, it will throw this error.

    Returns
    -------
    output : array_like
            The data after the change.

    '''

    w = [float(i) / (float(fs) / 2) for i in fc] # Normalize the frequency
    
    if filter_design == 'butter': 
        b, a = signal.butter(order, w, filter_pass)
    elif filter_design == 'bessel':
        b, a = signal.bessel(order, w, filter_pass)
    elif filter_design == 'chebyii':
        b, a = signal.cheby2(order, 6, w, filter_pass)
    elif filter_design == 'fir':
        b = signal.filtwin(order, w, pass_zero = False)
        a = 1
    else:
        print(str(filter_design) + " not recognized")
        raise ValueError
        
    if zero_phase:
        return signal.filtfilt(b, a, inp)
    
    return signal.lfilter(b, a, inp)

def calc_fft(inp, fs, positive_only = False, plot = False):
    '''
    Calculates and returns the FFT of the data.

    Parameters
    ----------
    inp : array_like
        Signal to resample.
    fs : int
        Original sample rate.
    positive_only : bool, optional
        If true, only the positive frequencies of the FFT will be returned. The default is False.
    plt : bool, optional
        If true, the FTT will be plotted. The default is False.
    
    Returns
    -------
    freq : array_like
        The DST sample frequencies.
    signal_fx : array_like
        The FFT complex amplitude values.

    '''
    
    signal_fx = scipy.fft.fft(inp)
    freq = scipy.fft.fftfreq(len(signal_fx), d = 1/fs)
            
    #Truncate to positive frequencies only
    if(positive_only):
        n = int(len(inp) / 2)
        signal_fx = signal_fx[1:n]
        freq = freq[1:n]
        
    if(plot):
        plt.figure()
        plt.title("FFT")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(freq, abs(signal_fx) **2)
        plt.show()
    
    return freq, signal_fx
     

def trim_signal(inp, fs, lower_bound = 0, upper_bound = 0):
    '''
    Truncates the signal.

    Parameters
    ----------
    inp : array_like
            Signal to resample.
    fs : int
        Original sample rate.
    lower_bound : float, optional
        Where to start the signal, in seconds. The default is 0.
    upper_bound : float, optional
        Where to cut off the signal (original data), in seconds. The default is 0.
    
    Returns
    -------
    output : array_like
            The data after the change.

    '''
    lower_bound = int(lower_bound * fs)
    upper_bound = int(upper_bound * fs)        
    
    if upper_bound <= lower_bound or upper_bound >= len(inp):
        upper_bound = len(inp)
    
    
    return inp[lower_bound : upper_bound]
     

def shift(inp, fs, delay_amount):
    '''
    Function to shift the signal in time.

    Parameters
    ----------
    inp : array_like
            Signal to resample.
    fs : int
        Original sample rate.
    delay_amount : float
        The amount to delay the signal, in seconds. A positive amount will move up the signal, and a negative amount will truncate the end of the signal.

    Returns
    -------
    output : array_like
            The data after the change.

    '''
    
    delay_in_samples = int(delay_amount * fs)
    out = np.zeros(len(inp) - abs(delay_in_samples))
    
    if delay_in_samples >= 0:
        out = inp[delay_in_samples:]
    else:
        out = inp[0:delay_in_samples]
    
    return out
 
def invert(inp):
    '''
    Function to flip the waveform

    Parameters
    ----------
    inp : array_like
            Signal to resample.

    Returns
    -------
    output : array_like
            The data after the change.

    '''
    
    out = -np.asarray(inp)
        
    return out

def moving_average(inp, N):
    '''
    Function that returns a moving average of the input signal.

    Parameters
    ----------
    inp : array_like
        Signal to filter.
    N : int
        Window size.

    Returns
    -------
    out : array_like
        The filtered signal.

    '''
    
    out = np.convolve(inp, np.ones(N)/N, mode='valid')
    
    return out

def split_by_event_markers(signal, event_markers):
    '''
    Split a BioSignal into smaller ones based on event markers

    Parameters
    ----------
    signal : BioSignal
        BioSignal to split.
    event_markers : array_like
        When to split the BioSignal.

    Returns
    -------
    array_like
        Partitioned BioSignals.

    '''
    
    output = []
    data = copy.deepcopy(signal.data)
    
    last_event_marker_t = 0
    for i in range (0, len(event_markers)+1):
        #First, instantiate a new copy of the signal.
        new_sig = copy.deepcopy(signal)
        
        #Calculate the endpoint. If this is there are no more event markers, don't truncate the end
        if i < len(event_markers):
            cur_event_marker_t = int(event_markers[i].t * signal.fs)
        else:
            cur_event_marker_t = len(data)
        
        #Segment the output by the event markers
        new_sig.data = data[last_event_marker_t: cur_event_marker_t]
        
        #Update the last event marker time
        last_event_marker_t = cur_event_marker_t
        
        #Finally, add this copy to to the output list
        output.append(new_sig)
        
    #Convert the list to array and return
    return np.asarray(output)    
    

def align_signals(signal1, signal2, tfs = 1000):
    #Make deep copies of the signals
    sig1 = copy.deepcopy(signal1)
    sig2 = copy.deepcopy(signal2)
    
    #Time resolution between signals needs to be the same.
    sig1.resample(tfs)
    sig2.resample(tfs)         
    
    sigplot.plot_biosignals([sig1, sig2])
    
    
    #Cross-correlate
    corr = signal.correlate(sig1.data, sig2.data, method = 'fft')
    
    print(len(corr))
    print(corr)
    
    plt.figure()
    plt.plot(corr)

       
    return signal1, signal2
