from scipy.signal import find_peaks
import numpy as np
from ppgtools import sigpro
import copy
import matplotlib.pyplot as plt

import heartpy as hp
from ecgdetectors import Detectors

def find_peaks_heartpy(inp, fs, plot = False):
    working_data, measures = hp.process(inp, fs)
    
    if plot:
        plt.figure()
        hp.plotter(working_data, measures)
        
    return working_data, measures

def get_peak_frequency(inp, fs, fc = [], padlen = 0):
    sig = copy.deepcopy(inp)
    
    #Pad the signal
    sig = np.append(sig, [0] * padlen)
    
    #If cutoffs are provided, filter the signal
    if len(fc) != 0:
        if len(fc) != 2:
            raise ValueError("There needs to be two cutoff frequencies, as this uses a bandpass.")
        
        sig = sigpro.filter_signal(sig, fs, "Bandpass", fc)
        
    #Calculate the frequency response of the signal    
    plot = True
    plot = False
    freq, fft = sigpro.calc_fft(sig, fs, True, plot)
    
    #Obtain the max frequency
    max_freq = freq[np.argmax(fft)]
    
    return max_freq
    
def find_mins_and_peaks_adaptive_threshold(inp, fs, N, fc = None, plot = False):
    '''
    Function to determine the location of peaks using an adaptive threshold.
    The adaptive threshold is obtained from the moving average of the original signal.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks in.
    N : int
        Moving average size.
    fc : float, optional
        Cutoff frequencies for interesection calculation. The default is no filtering.
    plot : bool, optional
        Whether to plot the adaptive threshold or not. The default is false.

    Returns
    -------
    peaks : array_like
        Indices of the estimated peaks.

    '''
    peaks = []
    mins = []
    
    filt_sig = copy.deepcopy(inp)
    
    if fc != None:
        filt_sig = sigpro.filter_signal(filt_sig, fs, 'bandpass', fc)
    
    if N == 0:
        thresh = np.zeros(len(filt_sig.data))
    else:
        thresh = sigpro.moving_average(filt_sig, N)

    #thresh = sigpro.filter_signal(inp, fs, "lowpass", [1])

    start = int(N/2)
    end = N - start - 1
    if end > 0:
        intersections = np.argwhere(np.diff(np.sign(filt_sig[start:-end] - thresh))).flatten()
    else:
        intersections = np.argwhere(np.diff(np.sign(filt_sig[start:] - thresh))).flatten()
    
    #Determine if the first crossing is positive or negative (i.e., does the signal start above or below the threshold)
    starts_positive = thresh[0] > filt_sig[0]
    
    if starts_positive:
        i = 3
    else:
        i = 2
    while(i < len(intersections)):
        local_peak = np.argmax(inp[intersections[i-1]:intersections[i]])
        local_min = np.argmin(inp[intersections[i-2]:intersections[i-1]])
        peaks.append(local_peak + intersections[i-1])
        mins.append(local_min + intersections[i-2])
        i+=2
    
    peaks = np.asarray(peaks)
    mins = np.asarray(mins)
    
    if plot:
        plt.figure(figsize = [8.8, 4.8])
        plt.subplot(1,2,1)
        plt.plot(np.linspace(0, len(inp) / fs, len(inp)), inp, label = "Raw Signal")
        plt.plot(peaks / fs, inp[peaks], 'r.', label = "Peaks")
        plt.plot(mins / fs, inp[mins], 'b.', label = "Mins")
        
        for x in intersections:
            #plt.axvline(x)
            pass
        
        plt.subplot(1,2,2)
        plt.plot(np.linspace(0, len(filt_sig) / fs, len(filt_sig)), filt_sig, label = "Filtered Signal")
        plt.plot(np.linspace(0, len(thresh) / fs, len(thresh)), thresh, label = "Adaptive Threshold")
        
        plt.plot(peaks / fs, filt_sig[peaks], 'r.')
        plt.plot(mins /fs, filt_sig[mins], 'b.')
        
        plt.legend()
        plt.tight_layout()
    
    return mins, peaks

#####################################
# Waveform foot detection           #    
#####################################
def find_feet_intersecting_tangents(inp, peaks):
    feet = []
    
    plt.figure()
    plt.plot(inp)
    
    for i in range(1, len(peaks)):
        #For every peak, find the corresponding foot, so segment by peaks
        wave_seg = inp[peaks[i-1]:peaks[i]]
        wave_seg_d1 = np.diff(wave_seg)
        
        #First, find the amplitude of the minimum
        min_amp = min(wave_seg)
        
        #Next, find the location and amplitude of the maximum slope
        max_slope = max(wave_seg_d1)
        max_slope_loc = np.argmax(wave_seg_d1)
        max_slope_amp = wave_seg[max_slope_loc]
        
        #Find the intersection between the two tangents.
        #x2 = (y2 - y1 + mx1) / m
        foot_loc = (min_amp - max_slope_amp) / max_slope + max_slope_loc + peaks[i-1]
        feet.append(foot_loc)
        
        '''
        plt.scatter(max_slope_loc + peaks[i-1], max_slope_amp, color = 'r')
        plt.scatter((min_amp - max_slope_amp) / max_slope + max_slope_loc + peaks[i-1], min_amp, color = 'g')
        plt.scatter(peaks[i], inp[peaks[i]], color = 'b')
        plt.plot([foot_loc, max_slope_loc + peaks[i-1]],[min_amp, max_slope_amp], 'r--')
        plt.axhline(min_amp, color = "r", linestyle = '--')
        plt.xlabel("Samples [n]")
        plt.ylabel("Amplitude [a.u.]")
        
        plt.show()
        '''
        
        
    
    return np.asarray(feet)
    
def find_feet_sdm(inp, peaks): 
    '''
    Determine the foot of the wave through the second derivative maximums.

    Parameters
    ----------
    inp : array_like
        Signal to find feet.
    peaks : array_like
        Location (indices) of systolic peaks.

    Returns
    -------
    feet : array_like
        Location (indices) of waveform feet.

    '''
    feet = []
    d2 = np.diff(inp, n = 2)
    
    feet.append(np.argmax(d2[0:peaks[0]]))
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(inp)
    plt.ylabel("Amp. [a.u.]")
    plt.title("Signal")
    plt.xticks([])
    plt.subplot(2,1,2)
    plt.ylabel("Amp. [a.u.]")
    plt.plot(d2)
    plt.title("Second derivative")
    plt.xlabel("Samples [n]")
    
    for i in range(1, len(peaks)):
        #For every peak, find the corresponding SDM
        foot_loc = np.argmax(d2[peaks[i-1]:peaks[i]]) + peaks[i-1]
        plt.subplot(2,1,1)
        plt.scatter(foot_loc, inp[foot_loc], color = 'r')
        plt.subplot(2,1,2)
        plt.scatter(foot_loc, d2[foot_loc], color = 'r')
        feet.append(foot_loc)
    
    
    
    return np.asarray(feet)

def find_feet_local_min(inp, peaks):
    mins = []
    
    #First foot will probably be wrong, but just include it to keep the
    #list lengths the same.
    mins.append(np.argmin(inp[0:peaks[0]]))
    
    for i in range (1, len(peaks)):
        mins.append(np.argmin(inp[peaks[i-1]:peaks[i]]) + peaks[i-1])
        
    mins = np.array(mins)
    return mins
    
#####################################
# Peak detection using ecgdetectors #    
#####################################
def find_peaks_hamilton(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of P.S. Hamilton, “Open Source ECG Analysis Software Documentation”, E.P.Limited, 2002.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    detectors = Detectors(fs)
    return detectors.hamilton_detector(inp)
    
def find_peaks_christov(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of Ivaylo I. Christov, “Real time electrocardiogram QRS detection using combined adaptive threshold”, BioMedical Engineering OnLine 2004, vol. 3:28, 2004. 

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    detectors = Detectors(fs)
    return detectors.christov_detector(inp)

def find_peaks_engzee(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of W. Engelse and C. Zeelenberg, “A single scan algorithm for QRS detection and feature extraction”, IEEE Comp. in Cardiology, vol. 6, pp. 37-42, 1979 with modifications A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, “Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics”, BIOSIGNALS 2012, pp. 49-54, 2012.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    detectors = Detectors(fs)
    return detectors.engzee_detector(inp)

def find_peaks_pan_tompkins(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of Jiapu Pan and Willis J. Tompkins. “A Real-Time QRS Detection Algorithm”. In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    
    detectors = Detectors(fs)
    return detectors.pan_tompkins_detector(inp)

def find_peaks_swt(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation based on Vignesh Kalidas and Lakshman Tamil. “Real-time QRS detector using Stationary Wavelet Transform for Automated ECG Analysis”. In: 2017 IEEE 17th International Conference on Bioinformatics and Bioengineering (BIBE). Uses the Pan and Tompkins thresolding method.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    
    detectors = Detectors(fs)
    return detectors.swt_detector(inp)

def find_peaks_two_average(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). “Frequency Bands Effects on QRS Detection” The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    
    detectors = Detectors(fs)
    return detectors.two_average_detector(inp)

def find_peaks_distance(inp, distance):
    '''
    Simple peak detection using scipy.signal.find_peaks.
    The minimum distance between peaks must be supplied.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    distance : int
        Number of indices that each peak must be seperated by.

    Returns
    -------
    peaks : array_like
        Index locations of the peaks.

    '''
    peaks, _ = find_peaks(inp, distance = distance)
    return np.array(peaks)

def align_peaks(peaks1, peaks2):
    closest2 = []
    
    out1 = []
    out2 = []

    #Assumes same sample rate between peaks.
    #Loop through first array. Find the closest corresponding peak from peaks2 
    for i in peaks1:
        array = np.asarray(peaks2)
        idx = (np.abs(array - i)).argmin()
        closest2.append(idx)
        
    for i in range (0, len(peaks2)):
        val = peaks2[i]
        array = np.asarray(peaks1)
        idx = (np.abs(array - val)).argmin()
                
        #Check if they match.
        if(i == closest2[idx]):
            #out1.append(peaks1[idx])
            #out2.append(peaks2[i])
            out1.append(idx)
            out2.append(i)
            pass
            
    return np.array(out1), np.array(out2)
    #return closest1, closest2
        
    
    
    
    