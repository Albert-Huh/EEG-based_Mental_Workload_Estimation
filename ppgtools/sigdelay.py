import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cmath
import statsmodels.api as sm
from ppgtools import sigpeaks, sigpro, sigplot
import ppgtools
import copy

#Constants
col_width = 16
calibration_fs = 20

#Options for what type of algorithm will be used
use_xcorr = False
use_phase_slope = True



class Delay_Extractor():
    #Debug text
    show_delays = False
    show_xcorr = False
    show_phase_slope = False

    #Initialize a Delay_Extractor object. Here, we can set debug text, etc.
    def __init__(self):
        pass
    
    def delay_crosscorrelation(self, proximal_signal, distal_signal):
        pass
    
    def delay_intersecting_tangent(self, proximal_signal, distal_signal):
        prox_feet = proximal_signal.find_feet_intersecting_tangents()
        dist_feet = distal_signal.find_feet_intersecting_tangents()
        
        ptt = prox_feet - dist_feet
        print(ptt)
        
        return ptt
    
    def delay_phase_slope(self, proximal_signal, distal_signal, padlen = 0, use_hamming = False):
        '''
        Function to determine the delay between two signal segments through the phase slope method.

        Parameters
        ----------
        proximal_signal : BioSignal
            BioSignal containing the proximal signal.
        distal_signal : BioSignal
            BioSignal containing the distal signal.
        padlen : int, optional
            If the signals should be zero padded to increase frequency resolution. The default is no padding.
        use_hamming : bool, optional
            If a Hamming window should be used when calculating the FFTs. The default is false.
        Raises
        ------
        ValueError
            The two signals must have the same length.

        Returns
        -------
        delay : float
            The time delay of the two signals, in seconds.

        '''
        if len(proximal_signal.data) != len(distal_signal.data):
            print("Proximal (%d) and distal (%d) signals do not have the same length.", len(proximal_signal.data), len(distal_signal.data))
            raise ValueError
            
        #Make copies of the signals
        prox_copy = copy.deepcopy(proximal_signal)
        dist_copy = copy.deepcopy(distal_signal)
        
        #Apply Hamming Window if needed
        if use_hamming:
            prox_copy.data *= np.hamming(len(prox_copy.data))
            dist_copy.data *= np.hamming(len(dist_copy.data))
           
        #Add padding if needed    
        padding = [0] * padlen
        prox_copy.data = np.append(prox_copy.data, padding)
        dist_copy.data = np.append(dist_copy.data, padding)   
                        
        #Get the FFTs of both signals
        freq, fft_p = prox_copy.calc_fft()        
        freq, fft_d = dist_copy.calc_fft()
        
        #Find the FFT ratio
        fft_ratio = np.divide(fft_d, fft_p)
        
        #Get the phase values of each frequency from the FFT ratio
        fft_ratio_phases = []
        for i in range (0, len(fft_ratio)):
            fft_ratio_phases.append(cmath.phase(fft_ratio[i]))
        
        #Unwrap, in case there is phase ambiguity
        fft_ratio_phases = np.unwrap(fft_ratio_phases)
        
        #Find the slope through linear regression
        res_wls = sm.WLS(fft_ratio_phases, freq, weights = (abs(fft_p) * abs(fft_d))**2).fit()
        
        #Find the delay
        delay = (-res_wls.params[0] / (np.pi * 2))
        
        if False:
            #print(res_wls.summary())
            plt.figure(figsize = [12.8, 4.8])
            plt.subplot(2, 2, 1)
            plt.title("DFT")
            plt.ylabel("Proximal Signal\nAmplitude")
            plt.plot(freq, abs(fft_p))
            
            plt.subplot(2, 2, 3)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Distal Signal\nAmplitude")
            plt.plot(freq, abs(fft_d))
            
            plt.subplot(1, 2, 2)
            plt.title("Phase Slope")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Phase (rads)")
            # plt.scatter(freq, fft_ratio_phases)
            
            mag = (abs(fft_p)*abs(fft_d))**0.5
            mag  = np.round(mag / max(mag) * 4)/4
            rgba = np.zeros((len(freq), 4))
            rgba[:,3] = mag
            
            print(max(mag))
            print(len(rgba))
            print(len(freq))
            plt.scatter(freq, fft_ratio_phases, c = rgba)
            plt.plot(freq, res_wls.fittedvalues, 'g--', label="WLS, r = {0:.3g}, delay = {1:.3}ms".format(res_wls.rsquared, delay * 1000))
            plt.legend()
            plt.show()
            
        
        return delay
    
    def delay_sdm(self, proximal_signal, distal_signal):
        pass
    
    def delay_peaks(self, proximal_signal, distal_signal):
        '''
        Find the time delay of two signals by the time offset of the peaks.

        Parameters
        ----------
        proximal_signal : BioSignal
            BioSignal containing the proximal signal.
        distal_signal : BioSignal
            BioSignal containing the distal signal.

        Returns
        -------
        prox_peaks : array_like
            Indices of the peaks in the proximal signal.
        dist_peaks : array_like
            Indices of the peaks in the distal signal.
        ptt : array_like
            The transit times associated with each peak.

        '''
        #Make a deep copy
        prox_sig = copy.deepcopy(proximal_signal)
        dist_sig = copy.deepcopy(distal_signal)
        
        #Overfilter
        prox_sig.filter_signal("bandpass", [0.5, 3])
        dist_sig.filter_signal("bandpass", [0.5, 3])
        
        #Find peaks
        prox_peaks = prox_sig.find_peaks_adaptive_threshold(2)
        dist_peaks = dist_sig.find_peaks_adaptive_threshold(2)
        
        sigplot.plot_biosignals([prox_sig, dist_sig], peaks = [prox_peaks, dist_peaks])
        
        
        #Find the time difference between the two peaks
        ptt = np.subtract(dist_peaks, prox_peaks)
        
        #Convert to seconds
        ptt = ptt / prox_sig.fs
        
        return prox_peaks, dist_peaks, ptt
