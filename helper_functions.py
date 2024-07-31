import numpy as np
from scipy.signal import find_peaks

def upload_data(path2file, file_ending, calib='res50'):
    with open(path2file + 'data' + file_ending, 'rb') as f:
        raw = np.load(f)
    with open(path2file + 'lst' + file_ending, 'rb') as f:
        lst = np.load(f)
    with open(path2file + 'systime' + file_ending, 'rb') as f:
        systime = np.load(f)
    with open(path2file + 'short' + file_ending, 'rb') as f:
        short = np.load(f)
    with open(path2file + 'res50' + file_ending, 'rb') as f:
        res50 = np.load(f)
    with open(path2file + 'res100' + file_ending, 'rb') as f:
        res100 = np.load(f)
        
    if calib=='res50':
        return (raw - short) / (res50 - short), lst, systime
    elif calib=='res100':
        return (raw - short) / (res100 - short), lst, systime
    elif calib=='short':
        return raw - short, lst, systime
    elif calib=='raw':
        return raw, lst, systime
    

### binning functions ###

def lst_binning(data, lst, binsize, method='mean'): 
    binsize /= 60
    lst_bins = np.arange(0, 24, binsize)
    
    bin_inds = np.digitize(lst, bins=(lst_bins + binsize/2))
    bin_inds[np.where(bin_inds == len(lst_bins))[0]] = 0
    if method == 'mean':
        data_binned = np.array([np.nanmean(data[bin_inds == i], axis=0) for i in range(len(lst_bins))])   
    elif method == 'median':
        data_binned = np.array([np.nanmedian(data[bin_inds == i], axis=0) for i in range(len(lst_bins))])   
    return data_binned, lst_bins, bin_inds


def freq_binning(data, freq, binsize, flow, fhigh, method='mean'): 
    bins = np.arange(flow, fhigh, binsize)
    bin_inds = np.digitize(freq, bins=(bins + binsize / 2)) 
    if method == 'mean':
        data_binned = np.array([np.nanmean(data[:, bin_inds == i], axis=1) for i in range(len(bins))]).T   
    return data_binned, bin_inds


def binning(data, freq, lst, binsize_f, binsize_t, flow, fhigh): 
    """
    LST & freq binning
    """
    bins_f = np.arange(flow, fhigh, binsize_f)
    bin_inds_f = np.digitize(freq, bins=(bins_f + binsize_f / 2)) 
    
    binsize_t /= 60
    bins_t = np.arange(0, 24, binsize_t)
    bin_inds_t = np.digitize(lst, bins=(bins_t + binsize_t / 2)) 
    bin_inds_t[np.where(bin_inds_t == len(bins_t))[0]] = 0
    
    data_binned = np.array([[np.nanmean(data[bin_inds_t == i][:, bin_inds_f == j]) \
                             for j in range(len(bins_f))] for i in range(len(bins_t))])
    return data_binned, bins_f, bins_t, bin_inds_f, bin_inds_t


def truncate(data, freq, highpass, lowpass):
        '''
        Crop to frequency range of [highpass, lowpass).
        
        Parameters:
            highpass, lowpass (float): min/max frequency in MHz
        '''
        data = data[:, (freq >= highpass) & (freq < lowpass)]
        freq = freq[(freq >= highpass) & (freq < lowpass)]
        return data, freq

### other functions

def f2i(f, flow=30, fhigh=200, num_inds=2785):
    """convert freq to index"""
    return int((f-flow) * (num_inds/(fhigh-flow)))

def lst2ind(t, binsize):
    """convert lst to index for lst binned data"""
    binsize /= 60
    return int(t/binsize)


def split_days(lst):
    '''Split days by LST'''
    days = find_peaks(lst)[0]
    days = np.append(days, len(lst)-1)
    days += 1
    days = np.insert(days, 0, 0)
    return days
