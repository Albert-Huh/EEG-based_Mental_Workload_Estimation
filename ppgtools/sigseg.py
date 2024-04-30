from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import copy

col_width = 16

'''///////////////////Noisy segment classification//////////////////'''
def reject_data_threshold(t, metrics, threshold, threshold_type = 'outside'):
    '''
    Function to reject data that is impossible based on a threshold.
    The threshold is not inclusive (e.g., > not >=)

    Parameters
    ----------
    threshold : array_like or int
        The bounds of the threshold.
    threshold_type : {'inside', 'outside', 'above', 'below'}, optional
        What type of threshold to use. The default is 'outside'.

    Returns
    -------
    clean_t : array_like
        Timestamps [s] of biometrics during clean regions.
    clean_metrics : array_like
        Biometrics during clean regions.

    '''
    low = np.min(threshold)
    high = np.max(threshold)
        
    if threshold_type == 'outside':
        bad_t = np.argwhere(np.logical_or(metrics < low, metrics > high))
    elif threshold_type == 'inside':
        bad_t = np.argwhere(np.logical_not(np.logical_and(metrics > low, metrics < high)))
    elif threshold_type == 'above':
        bad_t = np.argwhere(metrics > high)
    elif threshold_type == 'below':
        bad_t = np.argwhere(metrics < low)
    else:
        raise ValueError ("Unknown threshold type.")
    

    clean_t = copy.deepcopy(t)
    clean_metrics = copy.deepcopy(metrics)
    
    clean_metrics = np.delete(clean_metrics, bad_t)
    clean_t = np.delete(clean_t, bad_t)
    
    return clean_t, clean_metrics
    


def reject_segments(t, metrics, noisy_region):
    '''
    Function to reject biometric data (e.g., HR, SpO2) if it was calculated in
    a noisy region.

    Parameters
    ----------
    t : array_like
        Timestamps of where each biometric was calculated. NOTE: this should be
        in the same units as the noisy_region timestamps (e.g., seconds).
        To convert from indices to time, divide by the sample rate.
    metrics : array_like
        Biometric data.
    noisy_region : array_like
        Array of timestamps of regions to reject from.

    Returns
    -------
    clean_t : array_like
        Timestamps [s] of biometrics during clean regions.
    clean_metrics : array_like
        Biometrics during clean regions.

    '''
    clean_t = copy.deepcopy(t)
    clean_metrics = copy.deepcopy(metrics)
    
    #Get rid of peaks and BPM inside of noisy regions
    bad_t = []
    for i in range (0, len(clean_metrics)):
        for j in range (0, len(noisy_region)):
            if noisy_region[j][0] <= clean_t[i] <= noisy_region[j][1]:
                bad_t.append(i)    
    
    clean_metrics = np.delete(clean_metrics, bad_t)
    clean_t = np.delete(clean_t, bad_t)
    
    return list((clean_t, clean_metrics))

def classify_saturated_regions(signals, thresholds):
    '''
    Function that labels where a signal has saturated.

    Parameters
    ----------
    signals : array_like
        Array of BioSignals.
    thresholds : array_like
        Each element should be a pair of lower and upper thresholds.

    Returns
    -------
    sat_regions : array_like
        Timestamps (start, end) of saturated regions.

    '''
    sat_regions = []
    
    for s in signals:
        cur_sig = s.data
        #Find the intersection points of the signal and the lower threshold.
        thresh = np.zeros(len(cur_sig)) + thresholds[0]
        intersections = np.argwhere(np.diff(np.sign(cur_sig - thresh))).flatten()
        
        #If the signal starts below the threshold, the first intersection is the
        #start of the saturated region and vice versa
        if cur_sig[0] > thresholds[0]:
            start1 = intersections[::2]
            end1 = intersections[1::2]
            print("1")
        else:
            start1 = np.insert(intersections[1::2], 0, 0)
            end1 = intersections[::2]
            print("2")
        #If the there are more starts then ends, we need to add an endpoint.
        #I.e., the signal ends below threshold.
        if len(start1) > len(end1):
            end1 = np.append(end1, len(cur_sig) - 1)            
            
            
        thresh = np.zeros(len(cur_sig)) + thresholds[1]    
        intersections = np.argwhere(np.diff(np.sign(cur_sig - thresh))).flatten()
        #Do the same for the upper threshold.
        if cur_sig[0] < thresholds[1]:
            start2 = intersections[::2]
            end2 = intersections[1::2]
            print("3")
        else:
            start2 = np.insert(intersections[1::2], 0, 0)
            end2 = intersections[::2]        
            print("4")
        if len(start2) > len(end2):
            end2 = np.append(end2, len(cur_sig) - 1)            
        
        start = np.concatenate((start1, start2)) / s.fs
        end = np.concatenate((end1, end2)) / s.fs  
        
        
        sat_region = np.transpose(np.vstack((start, end)))
        sat_regions.append(sat_region)
    
    return sat_regions    
    
def classify_noisy_segment_fft(signals, window_size = 6, threshold = 4, fc = None):
    '''
    Function to classify signal segments that should be discarded for noise.
    This function assumes that the signal is periodic with a dominant
    frequency. If any frequency content amplitudes are within a set threshold
    of the main frequency, the segment is labeled as noisy. A sliding window 
    of the signal is taken, and for each window the FFT is calculated.

    Parameters
    ----------
    signals : array_like
        Array of BioSignals.
    window_size : float, optional
        The window size in seconds. The default is 6
    threshold : float, optional
        The ratio of second frequency amplitude to main frequency amplitude to 
        classify the signal as noisy. The default is 4.
    fc : array_like, optional
        The cutoff frequencies to filter the signal at. The default is none.

    Returns
    -------
    noisy_regions : array_like
        Array of arrays. Each subarray contains the segments to be rejected.

    '''
    
    noisy_regions = []  #List of lists. Each sublist is the interval to reject.
    
    #Loop for every signal.
    for sig in signals:
        noisy_region = []
        t = window_size
        sig_filt = copy.deepcopy(sig)
        if fc != None:
            sig_filt.filter_signal("bandpass", fc)
    
        while t < len(sig_filt.data) / sig.fs:
            win = copy.deepcopy(sig_filt)
            
            #Segment the data in a certain window size.
            win.trim_signal(t-window_size,t)
            
            #Calculate the frequency content of the signal
            freq, fft = win.calc_fft(True, False)
            
            #Find the peaks
            freq_peaks_locs,__ = signal.find_peaks(abs(fft))
            
            #Find the amplitude of the peaks
            freq_peaks = fft[freq_peaks_locs]
            
            #Get the largest amplitude
            amp_max = np.amax(abs(freq_peaks))
            
            #Get the second largest amplitude
            freq_peaks2 = np.delete(freq_peaks, np.argmax(abs(freq_peaks)))
            
            #Check if the second frequency peak a certain precentage of the 
            #first peak. If so, label the region as noisy.
            if len(freq_peaks2) == 0:
                noisy_region.append([t, t - window_size])
            else:
                amp_second_max = np.amax(abs(freq_peaks2))
                    
                if abs(amp_max / amp_second_max)**2 < threshold: 
                    noisy_region.append([int(t-window_size), int((t))])
                
            t += window_size
            
        noisy_regions.append(noisy_region)
    
    return noisy_regions

def classify_disconnected_segments(markers):
    '''
    Function to classify signal segments that should be discarded since they
    were during a period of disconnection. This assumes the signal was padded
    to account for the disconnection.

    Parameters
    ----------
    markers : array_like
        Array of EventMarker. This should contain the disconnection events.

    Returns
    -------
    noisy_segments : array_like
        Pair of timepoints (start and end) representing signal that should be
        discarded.

    '''
    noisy_region = []
    
    disc_t = 0    
    for e in markers:
        if "disconnected" in e.label:
            disc_t = e.t
        elif "reconnected" in e.label:
            noisy_region.append([disc_t, e.t])
    
    return noisy_region

'''////////////////////Waveform Segmentation/////////////////////////'''

    