import csv
import numpy as np
import os

def save_biometric_data(data, filename, loc = '', headers = []):
    '''
    Function to save extracted biometric data, such as HR or PWV

    Parameters
    ----------
    data : matrix
        Each column of the matrix should be one biometric.
    filename : string
        Name of the file.
    loc : string, optional
        Path to save the file. The default is in the ppgtools folder.
    headers : string, optional
        This will be the first row of the CSV. Put the name of the biometric. The default is "Unlabeled".
        This needs to be the same length as the number of columns, or it will revert to the default.

    Returns
    -------
    None.

    '''
    if np.any(data == None):
        print("No data to save.")
        return
    print(loc + '\\' + filename + '.csv')
    with open(loc + '\\' + filename + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        if len(headers) != np.shape(data)[1]:
            headers = ["Unlabeled" for i in range(0, len(data))]
            
        writer.writerow(headers)    
        writer.writerows(data)
        
def load_biometric_data(loc, filename = ''):
    filenames = []
    
    #If a filename is given, use that.
    if filename != '':
        filenames = [filename]
    else:        
        filenames = os.listdir(loc)
    
    
    headers = []
    all_data = np.zeros(0)
    sep_data = []
    
    for f in filenames:
        print(f)
        
        #Open the file
        with open(loc + '\\' + f, newline='') as f:
            reader = csv.reader(f)
            data = np.array(list(reader))
            
        headers = data[0]
        
        
        if len(all_data) == 0:
            all_data = data[1:]
        else:
            all_data = np.concatenate((all_data, data[1:]))
            
        sep_data.append(np.transpose(data[1:]).astype(np.float))
            
        
            
    all_data = np.transpose(all_data).astype(np.float)
    
    
            
    return headers, all_data, sep_data    
    
    
    '''
    
    

    headers = data[0]
    return headers, data[1:]
    
    '''