import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

class RFI_flagging:
    def __init__(self, data_product, lst_times):
        self.data = data_product
        self.lst = lst_times
        self.freq = np.linspace(0, 250, 4096)
    
    def __call__(self, hpass=30, lpass=200, binsize1=10, binsize2=5, thresh1=4, thresh2=4):
        self.truncate(hpass, lpass)
        self.rfi_remove(binsize1, thresh1)
        self.discard_bad_spectra()
        self.rfi_remove(binsize2, thresh2)

    def truncate(self, highpass, lowpass):
        flow = f2i(highpass, flow=0, fhigh=250, num_inds=4096)
        fhigh = f2i(lowpass, flow=0, fhigh=250, num_inds=4096)
        self.data = self.data[:, flow:fhigh]
        self.freq = self.freq[flow:fhigh]

    def rfi_remove(self, binsize, threshold):
        lst_binned, data_binned, bin_inds = lst_binning(self.data, self.lst, binsize, method='median')
        data_binned = median_filter(data_binned, size=(1,8))
        MAD = np.array([np.nanmedian(np.abs(data_binned[i] - self.data[bin_inds==i]), \
                                                   axis=0) for i in range(len(lst_binned))])
        MAD = median_filter(MAD, size=(1,80))
        x = (np.abs(self.data - data_binned[bin_inds]) > threshold * MAD[bin_inds])
        self.data[x] = np.nan

    def discard_bad_spectra(self):
        mask_rate_per_spectra = np.sum(np.isnan(self.data[:, f2i(50):]), axis=1)
        self.data[mask_rate_per_spectra > 0.2 * len(self.freq[f2i(50):])] = np.nan
    
    def discard_bad_days(self):
        days = split_days(self.lst)
        mask_rate_per_day = [np.sum(np.isnan(self.data[days[i]:days[i+1]])) / 
                             ((days[i+1] -days[i])*self.data.shape[1]) for i in range(len(days)-1)]
        for n, mask_rate in enumerate(mask_rate_per_day):
            if mask_rate > 0.5:
                self.data[days[n]:days[n+1]] = np.nan


def lst_binning(data, lst, binsize, method='mean'): 
    binsize /= 60
    bins = np.arange(binsize / 2, 24 + binsize / 2, binsize)
    bin_inds = np.digitize(lst, bins=(bins + binsize / 2)) 
    data_binned = np.zeros((len(bins), data.shape[1]))
    if method == 'mean':
        for i in range(len(bins)):
            data_binned[i] = np.nanmean(data[bin_inds == i], axis=0)  
    elif method == 'median':
        for i in range(len(bins)):
            data_binned[i] = np.nanmedian(data[bin_inds == i], axis=0) 
    return bins, data_binned, bin_inds


def f2i(f, flow=30, fhigh=200, num_inds=2785):
    """convert freq to index"""
    return int((f-flow) * (num_inds/(fhigh-flow)))


def lst2ind(t, binsize):
    """convert lst to index for lst binned data"""
    binsize /= 60
    return int(t/binsize)


def split_days(lst):
    days = find_peaks(lst)[0]
    days = np.append(days, len(lst)-1)
    days += 1
    days = np.insert(days, 0, 0)
    return days
