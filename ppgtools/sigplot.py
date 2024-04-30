import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

#This package is used for plotting multiple BioSignals.    
def plot_biosignals(signals, event_markers = None, points_of_interest = None, intervals = None):
    '''
    Function to plot biosignals.

    Parameters
    ----------
    signals : array_like of BioSignal
        Array of BioSignals to plot.
    event_markers : array_like of EventMarker, optional
        Add if event markers should be plotted.
    peaks : array_like, optional
        Array of arrays. Each inner array is the locations of peaks to be plotted. The function will automatically plot the correct amplitude. 
        The outer array length must be the same length as the number of signals.
        If a signal does not need points plotted, input an empty inner array.
    intervals: array_like, optional
        Array of arrays. Each inner array contains intervals to highlight in red.
        The outer array length must be the same length as the number of signals.
        If a signal does not need regions highlighted, input an empty inner array.
    Returns
    -------
    None.

    '''
    
    num_ax = len(signals)
    if event_markers != None:
        num_ax +=1
    
    fig, axs = plt.subplots(num_ax, 1, sharex = True)
    fig.subplots_adjust(hspace=0)
    
    for i in range (0, len(signals)):
        t = np.linspace(0, len(signals[i].data) / signals[i].fs, len(signals[i].data))
        
        if hasattr(axs, "__getitem__"):
            ax = axs[i]
        else:
            ax = axs
        
        #h = ax.set_ylabel(signals[i].name)  
        
        
        ax.plot(t, signals[i].data, c = colors[i % len(colors)])
        ax.set_ylabel(signals[i].units)  
        #ax.set_yticks([])
        #ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.text(0.97,0.1, signals[i].name, horizontalalignment='right', transform=ax.transAxes)
            
        if event_markers != None:
            for j in range (0, len(event_markers)):
                ax.axvline(event_markers[j].t, c = 'red')
            
        #If there are peaks to plot
        if points_of_interest != None and len(points_of_interest) == len(signals):
            ax.plot(t[points_of_interest[i]], signals[i].data[points_of_interest[i]], 'ro')
        
        #If there are intervals to plot
        if intervals != None and len(intervals) == len(signals):
            for interval in intervals[i]:
                ax.axvspan(interval[0], interval[1], alpha=0.5, color='red')
        
    #Plot the event markers
    last_xpoint = -1
    y_point = 0
    if event_markers != None:
        for i in range (0, len(event_markers)):
            if last_xpoint == event_markers[i].t:
                y_point -= 100
            else:
                y_point = 0
                last_xpoint = event_markers[i].t
            axs[len(signals)].annotate(event_markers[i].label, [event_markers[i].t, y_point], rotation = 'vertical', size = 6)
            axs[len(signals)].axvline(event_markers[i].t, c = 'red')
            
            axs[len(signals)].set_yticklabels([])
            axs[len(signals)].set_yticks([])
            axs[len(signals)].spines['top'].set_visible(False)
            axs[len(signals)].spines['right'].set_visible(False)
            axs[len(signals)].spines['bottom'].set_visible(False)
            axs[len(signals)].spines['left'].set_visible(False)

    fig.legend()
    plt.xlabel("Time [s]")
    #plt.tight_layout()

def bland_altman_plot(ref_data, extracted_data, units = '', *args, **kwargs):
    for i in range(len(extracted_data)):
        d = extracted_data[i]
        good = ~np.logical_or(np.isnan(ref_data), np.isnan(d))
        r = ref_data[good]
        d = d[good]
        
        r         = np.asarray(r)
        d         = np.asarray(d)
        mean      = np.mean([r, d], axis=0)
        diff      = r - d                    # Difference between data1 and data2
        md        = np.mean(diff)                   # Mean of the difference
        sd        = np.std(diff, axis=0)            # Standard deviation of the difference
        print(sd)
    
        plt.figure(dpi = 200)
        plt.title("Bland-Altman Plot")
        plt.scatter(mean, diff, c = colors[i], *args, **kwargs, s = 0.1)
        plt.axhline(md,           color='gray', linestyle='--')
        plt.text(max(mean), md + 1.96*sd, "+1.96 SD", ha = 'right', va = 'bottom')
        plt.text(max(mean), md, "Mean Difference", ha = 'right', va = 'bottom')
        plt.text(max(mean), md - 1.96*sd, "-1.96 SD", ha = 'right', va = 'bottom')
        plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
        plt.xlabel("Mean " + units)    
        plt.ylabel("Difference " + units)
    
def correlation_plot(ref_data, extracted_data):
    r = []
    
    plt.figure(dpi = 200)
    for i in range(len(extracted_data)):
        d = extracted_data[i]
        m, b = np.polyfit(ref_data, d, 1)
        t = np.linspace(np.min(ref_data), np.max(ref_data), 2)
        plt.plot(t, t * m + b, '--', c = colors[i])
        
        good = ~np.logical_or(np.isnan(ref_data), np.isnan(d))
        r.append(stats.pearsonr(ref_data[good], d[good]))
        
    for i in range(len(extracted_data)):
        d = extracted_data[i]
        plt.plot(ref_data, d, '.', c = colors[i], markersize = 0.5)
    
    return r
