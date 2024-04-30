from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
import numpy as np

def unmix_ICA(sigs): 
    X = []
    for sig in sigs:
        X.append(sig.data)
    
    X = np.transpose(X)
    
    # Compute ICA
    ica = FastICA()
    S_ = ica.fit_transform(X)  # Reconstruct signals
    
    unmixed = np.transpose(S_)
    t = np.linspace(0, len(sigs[0].data) / sigs[0].fs, len(sigs[0].data))
  
    #plt.figure()
    plt.subplots(len(sigs), 2, sharex = True)
    for i in range(len(sigs)):
        plt.subplot(len(sigs), 2, 2 * i + 1)
        plt.plot(t, sigs[i].data)
        plt.subplot(len(sigs), 2, 2 * i + 2)
        plt.plot(t, unmixed[i])
    
    
    return unmixed