import numpy as np
from scipy.ndimage import median_filter  
from helper_functions import *

class RFI_flagging:
    def __init__(self, data, freq, lst, systime):
        self.data = data
        self.lst = lst
        self.freq = freq
        self.systime = systime
    
    def __call__(self, hpass, lpass, binsize, thresh1, window, thresh2, template, \
                 known_rfi, width, thresh3, thresh4, bad_days):
        """
        Note: this is slow (10-30min depending on instrument/channel). you can find already flagged data for 2021 on github.
        
        Parameters:
            template ([[freq, flag_inds]]):  list of freq, flag_ind pairs
                    flag_inds: flagging template (list of indices (1min binned) to flag)
                    freq: list of frequencies on which to apply flag_inds
            see corresponding function for other parameters.
        """
        self.truncate(hpass, lpass)
        self.lst_flagging(binsize, thresh1)
        self.spectral_flagging(window, thresh2)
        self.timestream_flagging()
        for t in template:
            self.template_flagging(t[0], t[1])
        self.flag_known_rfi(known_rfi, width)
        self.flag_bad_freq(thresh3)
        self.flag_bad_spectra(thresh4)
        self.flag_bad_days(bad_days)
        self.truncate(hpass+0.5, lpass-0.5)
        return self.data, self.freq

    def truncate(self, highpass, lowpass):
        '''
        Crop to frequency range of [highpass, lowpass).
        
        Parameters:
            highpass, lowpass (float): min/max frequency in MHz
        '''
        self.data = self.data[:, (self.freq >= highpass) & (self.freq < lowpass)]
        self.freq = self.freq[(self.freq >= highpass) & (self.freq < lowpass)]
        
    def lst_flagging(self, binsize, threshold):
        '''
        Flagging based on median absolute deviation from median lst binned data.
        
        Parameters:
            binsize (int): binsize in minutes for LST binning
            thresh (int): multiple of MAD for flagging threshold
        '''
        lst_binned, data_binned, bin_inds = lst_binning(self.data, self.lst, binsize, method='median')
        MAD = np.array([np.median(np.abs(data_binned[i] - self.data[bin_inds==i]), \
                                                   axis=0) for i in range(len(lst_binned))])
        MAD = median_filter(MAD, size=100)
        x = (np.abs(self.data - data_binned[bin_inds]) > threshold *  MAD[bin_inds])
        self.data[x] = np.nan

    def spectral_flagging(self, window, thresh):
        '''
        Flagging based on standard deviation filter across frequency space.
        
        Parameters:
            window (int): size of filter
            thresh (int): flagging threshold
        '''
        h = window // 2
        N = self.data.shape[1]
        
        # arange data for filter of size window across frequency
        x = np.array([self.data[:, i - h:i + (h+1)] for i in range(h, N - h)])
        # median filter
        med_filt = np.nanmedian(x, axis=-1)
        # detrend
        x -= med_filt[:,:,None]
        # build data for filter with central point removed
        x_flag = np.delete(x, h, axis=-1)
        
        # compute std filter with & without central point
        std = np.nanstd(x, axis=-1)
        flag_std = np.nanstd(x_flag, axis=-1)

        # relative standard deviation residuals
        res = ((std - flag_std) / med_filt).T
        # flag where residual > thresh
        flag_inds = np.where(res > thresh)

        # crop data (filter didn't compute std at edges) & apply flags
        self.data = self.data[:, h : N - h]
        self.freq = self.freq[h : N - h]
        self.data[flag_inds] = np.nan

    def timestream_flagging(self, window=10):
        '''
        Flag outliers in detrended data.
        
        Parameters:
            window (int): filter size for detrending median filter
        '''
        # splits data into chunks so large time jumps don't screw up median filter
        breaks = np.where(np.abs(np.diff(self.lst)) % 23.5 > 1)[0]
        breaks = np.delete(breaks, np.where(np.diff(breaks) == 1)[0]) + 1
        breaks = np.insert(breaks, 0, 0)
        breaks = np.append(breaks, len(self.lst))

        filt_inds = np.zeros(len(self.lst), dtype=int)
        bin0 = -1

        med_filts = []
        # bin each chunk of data
        for b in range(len(breaks)-1):
            splits = range(breaks[b], breaks[b+1], window)

            temp = np.nanmedian([self.data[splits[i]:splits[i+1]] for i in range(len(splits)-1)], axis=1)
            med_filts.append(np.vstack([np.nanmedian(self.data[splits[-1]:breaks[b+1]], axis=0), temp]))

            filt_inds[breaks[b]:breaks[b+1]] = np.digitize(range(breaks[b], breaks[b+1]), splits) + bin0
            bin0 = filt_inds[breaks[b+1]-1]
        med_filt = np.vstack(med_filts)
        # thresh > 3sigma*2
        thresh = np.nanquantile(np.abs(self.data - med_filt[filt_inds]), 0.997, axis=0)
        # flag if abs deviation > thresh
        self.data[np.abs(self.data - med_filt[filt_inds]) > thresh*2] = np.nan
    
    def template_flagging(self, freqs, template, day_1=1634844413, offset=18.75/60, num_days=100):
        '''
        Data has well defined temporally periodic dips in some frequency channels. 
        Apply a flagging template for these channels.
        
        Parameters:
            freqs (array): indices (int) of self.freq to apply template on
            template (array): indices of 1min binned LST to flag on each day (shifted by offset)
            day_1 (int): unix time of first occurance of LST = 0 (used to split days in unix time)
            offset (float): offset per day (hours)
            num_days (int): number of days for binning (anything greater than number of days spanned by data works)
        '''
        s_per_lst_day = 86164.0905
        # split data by days in unix time
        systime_day_splits = np.digitize(self.systime - day_1, bins=np.arange(0, s_per_lst_day*num_days, s_per_lst_day))
        # apply offset per day
        lst_shift = (self.lst + systime_day_splits * offset) % 24
        
        # bin shifted lst in 1 min bins
        binsize = 1/60
        bin_inds = np.digitize(lst_shift, bins=np.arange(binsize,24,binsize))
        # apply template
        for f in freqs:
            for t in template:
                self.data[bin_inds == t, f] = np.nan
    
    def flag_known_rfi(self, known_rfi, width):
        '''
        Flag entire frequency channel. Use for known sources of RFI, ie FPGA clock
        
        Parameters:
            known_rfi (array): frequencies (MHz) to flag
            width (float): bandwidth around known_rfi to flag (MHz)
        '''
        for f in known_rfi:
            self.data[:, (self.freq >= f - width/2) & (self.freq < f + width/2)] = np.nan
    
    def flag_bad_freq(self, thresh):
        '''
        Flag entire frequency channel with flag rate greater than thresh.
        
        Parameters:
            thresh (float): flag rate threshold
        '''
        flag_rate_per_freq = np.sum(np.isnan(self.data), axis=0)/self.data.shape[0]
        self.data[:, flag_rate_per_freq > thresh] = np.nan
        
    def flag_bad_spectra(self, thresh):
        '''
        Flag entire spectra with flag rate greater than thresh.
        
        Parameters:
            thresh (float): flag rate threshold
        '''
        flag_rate_per_spec = np.sum(np.isnan(self.data), axis=1)/self.data.shape[1]
        self.data[flag_rate_per_spec > thresh, :] = np.nan
    
    def flag_bad_days(self, bad_days):
        '''
        Flag entire days.
        
        Parameters:
            bad_days (array): day number (int) to flag
        '''
        days = split_days(self.lst)
        for i in bad_days:
            self.data[days[i]:days[i+1]] = np.nan
