import numpy as np
from scipy.signal import find_peaks

class DataBinning:
    def __init__(self, data_product, lst_times):
        self.data = data_product
        self.lst = lst_times
    
    def __call__(self, binsize1=60, binsize2=20, thresh1=5, thresh2=5, binsize3=2,):
        self.rfi_remove(binsize1, thresh1)
        self.discard_bad_days()
        self.rfi_remove(binsize2, thresh2)
        lst_binned, data_binned, _ = self.lst_bin(binsize3, method='mean')
        return lst_binned, data_binned
    
    def lst_bin(self, binsize, method='median'): 
        binsize /= 60
        bins = np.arange(binsize, 24 + binsize, binsize)
        bin_inds = np.digitize(self.lst, bins=bins) 

        lst_binned = (bins - binsize/2)[np.min(bin_inds):np.max(bin_inds)+1]
        data_binned = np.zeros((len(lst_binned), self.data.shape[1]))
        if method == 'median':
            for i in range(np.min(bin_inds),np.max(bin_inds)+1):
                data_binned[i - np.min(bin_inds)] = np.nanmedian(self.data[bin_inds == i], axis=0)
        elif method == 'mean':
            for i in range(len(lst_binned)):
                data_binned[i - np.min(bin_inds)] = np.nanmean(self.data[bin_inds == i], axis=0)           
        return lst_binned, data_binned, bin_inds

    def rfi_remove(self, binsize, threshold):
        lst_binned, data_binned, bin_inds = self.lst_bin(binsize)
        MAD = np.array([np.nanmedian(np.abs(data_binned[i] - self.data[bin_inds==i]), axis=0) \
                        for i in range(len(lst_binned))])
        x = (np.abs(self.data - data_binned[bin_inds]) > threshold * MAD[bin_inds])
        self.data[x] = np.nan
        return self.data
    
    def discard_bad_days(self):
        days = self.split_days()
        mask_rate_per_day = [np.sum(np.isnan(self.data[days[i]:days[i+1]])) / 
                             ((days[i+1] -days[i])*self.data.shape[1]) for i in range(len(days)-1)]
        for n, mask_rate in enumerate(mask_rate_per_day):
            if mask_rate > 0.5:
                self.data[days[n]:days[n+1]] = np.nan
        return self.data
    
    def split_days(self):
        days = find_peaks(self.lst)[0]
        days = np.insert(days, 0, 0)
        days = np.append(days, len(self.lst)-1)
        days = days + 1
        return days