import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cmath
import statsmodels.api as sm
from ppgtools.sigpro import calc_FFT, filter_signal
from ppgtools.sigimport import importBIN, importCSV
import ppgtools

#Constants
col_width = 16
calibration_fs = 20

#Options for what type of algorithm will be used
use_xcorr = False
use_phase_slope = True

#Debug text
show_delays = False
show_xcorr = False
show_phase_slope = False

'''////////////////////Time Delay Extraction//////////////////////////'''
def extract_ptt(proximal_sig, distal_sig, fs, window_size, window_overlap, calibration = []):
    print("Extracting PTT")
    starting_index = 0
    ending_index = window_size
    index_incr = int(window_size * (1-window_overlap))
    last_percent_written = -10
    
    PTT = []
    time = []
    
    if show_delays:
        print("Course Delay".center(col_width)+"|"+"Fine Delay".center(col_width) +"|"+"Total Delay".center(col_width)+"|"+"Location".center(col_width))
        print("=" * (4*col_width + 3))
    while(ending_index < len(distal_sig)):
        #Act like inputs coming into ADC (taking a segment of whole data)
        #Correct
        ADC0 = proximal_sig[starting_index:ending_index]
        ADC1 = distal_sig[starting_index:ending_index]
        
        #plot_waveforms(ADC0, ADC1, fs = fs)
                
        #Find time delay from FFT Ratio
        if use_phase_slope:
            course_delay, fine_delay, total_delay = tt_fft_ratio(ADC0, ADC1, window_size, fs, calibration)
        else:
            course_delay = tt_xcorr(ADC0, ADC1, show_xcorr)
            if(course_delay > window_size / 2):      
                course_delay = -(window_size - course_delay)
            
            course_delay = course_delay / fs * 1000
            
            fine_delay = 0
            total_delay = course_delay
        
        time.append(ending_index / fs)
        PTT.append(total_delay)
        
        if show_delays:
            print('{0:.3g} ms'.format(course_delay).center(col_width), end = "|")
            print('{0:.3g} ms'.format(fine_delay).center(col_width), end = "|")
            print('{0:.3g} ms'.format(total_delay).center(col_width), end = "|")
            print(str(ending_index / fs).center(col_width))
        
        #Increment the window
        starting_index += index_incr
        ending_index += index_incr
        
        percent_complete = ending_index / len(distal_sig) * 100
        
        if(int(percent_complete) % 10 == 0 and int(percent_complete) != last_percent_written):
            print(str(int(percent_complete)) + "% complete")
            last_percent_written = int(percent_complete)
        
    return PTT, time

'''////////////////Time delay in a window/////////////////////////////'''
def tt_xcorr(a, b, show_plot = False):
    nperseg = len(a)
    course_xcorr = abs(np.fft.ifft(np.fft.fft(a, n = nperseg) * np.fft.fft(b, n = nperseg).conj()))
    course_xcorr = course_xcorr
    peak_course_xcorr = np.argmax(course_xcorr)
    
    if show_plot:
        plt.title("Cross Correlation")
        plt.xlabel("Delay [n]")
        plt.ylabel("Cross Correlation Amplitude")
        plt.yticks([])
        plt.plot(course_xcorr)
        plt.plot(peak_course_xcorr, course_xcorr[peak_course_xcorr], "ro")
        plt.show()
        
    #print(peak_course_xcorr)
    return peak_course_xcorr

def tt_fft_ratio(ADC0, ADC1, nperseg, fs, calibration):
    #STEP 1: COARSE TIMING ESTIMATION BY CROSSCORRELATION    
    if use_xcorr:
        course_delay = tt_xcorr(ADC0, ADC1, show_xcorr)
    else:
        course_delay = 0
    
    #STEP 2: FINE ESTIMATION VIA SLOPE OF RELATIVE PHASE
    #Step 2a: Coarse correction
    #Shift signal by coarse correction and truncate other channel
    if (course_delay == 0):
        shifted_ADC0 = ADC0.copy()
        shifted_ADC1 = ADC1.copy()
        course_delay = 0
    else:
        if(course_delay > nperseg / 2):        
            #Delay Channel 1 
            #Made negative to indicate time advance
            course_delay = -(nperseg - course_delay)  #Circuluar xcorr, so negative delay rep. on other side of nperseg      
            shifted_ADC0 = ADC0[0:nperseg + course_delay]
            shifted_ADC1 = ADC1[-course_delay: nperseg]
            course_delay *= -1
            
        elif(course_delay <= nperseg / 2):
            #Delay Channel 0
            shifted_ADC0 = ADC0[course_delay: nperseg]
            shifted_ADC1 = ADC1[0:nperseg - course_delay]
            course_delay *= -1
        
    #Try padding the data for higher freq. resolution? And Window?
    shifted_ADC0 *= np.hamming(len(shifted_ADC0))
    shifted_ADC1 *= np.hamming(len(shifted_ADC1))
    padding = [0] * int(len(shifted_ADC0) * 1023)
    shifted_ADC0 = np.append(shifted_ADC0, padding)
    shifted_ADC1 = np.append(shifted_ADC1, padding)       
    
    #Step 2b: Weighted phase and Linear Regression
    #Here, since delay is within a single sample, we can assume no phase ambiguity
    freq, ADC0_freq = calc_FFT(shifted_ADC0, fs)
    freq, ADC1_freq = calc_FFT(shifted_ADC1, fs)
    
    #Send in calibration ratio
    if len(calibration) > 0:
        #Make sure it is the same frequency range
        #Frequency resolution = Fs/N, max frequency = fs/2
        #We want to change the max frequency from calibration_fs/2 to fs/2
        calibration = calibration[0:int(fs/2)]        
        
        #Resample the calibration so it matches the same frequency resolution
        calibration = signal.resample(calibration, len(freq))
    
    fine_delay = 1000 * phase_slope(freq, ADC0_freq, ADC1_freq, calibration) #in ms
    
    
    
    #Step 2c: Combined delay
    course_delay = course_delay / fs * 1000
    total_delay = course_delay + fine_delay 
    
    #Step 2d: Scaling factor
    #???????
    
    #STEP 3: RETURN VALUES
    return course_delay, fine_delay, total_delay


'''////////////////Helper Functions for FFT Ratio/////////////////////'''

def phase_slope(freq, signal_1_fx, signal_2_fx, calibration = []):    
    #Get FFT ratio
    fft_ratio = np.divide(signal_1_fx, signal_2_fx)        
                
    #Get the phase values of each frequency from the FFT ratio
    fft_ratio_phases = []
    for i in range (0, len(fft_ratio)):
        fft_ratio_phases.append(cmath.phase(fft_ratio[i]))
        
    if len(calibration) == len(fft_ratio):
        print("Calibrating...")
        fft_ratio_phases = [x - y for x,y in zip(fft_ratio_phases, calibration)]
    
    np.unwrap(fft_ratio_phases)
    #Weighted by average power of both frequency spectrums
    res_wls = sm.WLS(fft_ratio_phases, freq, weights = (abs(signal_1_fx)**2 * abs(signal_2_fx)**2)**2).fit()
        
    if show_phase_slope:
        #print(res_wls.summary())
        plt.figure(figsize = [12.8, 4.8])
        plt.subplot(2, 2, 1)
        plt.title("DFT")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Proximal Signal\nAmplitude")
        plt.plot(freq, abs(signal_1_fx))
        
        plt.subplot(2, 2, 3)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Distal Signal\nAmplitude")
        plt.plot(freq, abs(signal_2_fx))
        
        plt.subplot(1, 2, 2)
        plt.title("Phase Slope")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (rads)")
        # plt.scatter(freq, fft_ratio_phases)
        
        mag = (abs(signal_1_fx)**2*abs(signal_2_fx)**2)
        mag  = np.round(mag / max(mag) * 4)/4
        rgba = np.zeros((len(freq), 4))
        rgba[:,3] = mag
        
        print(max(mag))
        print(len(rgba))
        print(len(freq))
        plt.scatter(freq, fft_ratio_phases, c = rgba)
        plt.plot(freq, res_wls.fittedvalues, 'g--', label="WLS")
        
        plt.show()
    
    return(-res_wls.params[0] / (np.pi * 2))
    
# =============================================================================
# Load in hardware calibration data and returns the calibration array in frequency domain
# =============================================================================
def calibrate_hardware(loc):
    calibration = importCSV(loc, 2)
    plt.figure()
    plt.plot(calibration[0], calibration[1])
    
    return calibration[1]