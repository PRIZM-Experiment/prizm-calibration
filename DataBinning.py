import numpy as np
from scipy import ndimage


class DataBinning:

    def __init__(self, data_product, times, min_per_bin, freq_low, freq_high):
        self.data = data_product
        self.times = times
        self.min_per_bin = min_per_bin
        self.flow = freq_low
        self.fhigh = freq_high

    def __call__(self, threshold=200, median_filter_window=20):
        self.freq_binning()
        self.rfi_remove(threshold, median_filter_window)
        binned_data = self.antenna_binning()
        return binned_data

    def freq_binning(self):
        """ Sorts the antenna data into frequency bins matching the GSM (usually 2MHz) and cuts out
        frequencies that are higher or lower than GSM """
        freqlow = [int(4096*j/250) for j in range(self.flow - 1, self.fhigh + 1, 2)]
        freqhigh = [int(4096*j/250) for j in range(self.flow + 1, self.fhigh + 3, 2)]
        a = []
        for k in range(0, int((self.fhigh - self.flow) / 2) + 1):
            a.append(np.sum(self.data[:, freqlow[k]:freqhigh[k]], axis=1) / (freqhigh[k] - freqlow[k]))
        self.data = np.array(a).T
        return self.data

    def rfi_remove(self, threshold, median_filter_window):
        """ Removes RFI data from the antenna data """

        medians = np.median(self.data, axis=0)
        flattened = self.data - medians
        filtered = ndimage.median_filter(flattened, [1, median_filter_window])
        corrected = flattened - filtered
        MAD = np.median(np.abs(corrected - np.median(corrected)))
        x = (corrected - np.median(corrected) > threshold * MAD)
        self.data = np.ma.masked_array(self.data, x)
        return self.data

    def antenna_binning(self):
        freq_bins = int((self.fhigh - self.flow)/2) + 1
        minutes = self.times * 1440 / 23.94
        bins = int(1440 / self.min_per_bin)
        bin_num = np.zeros(len(minutes))
        for i, minute in enumerate(minutes):
            bin_num[i] = np.floor(minute / self.min_per_bin)
        freq_list = [i for i in range(0, freq_bins)]

        binned = np.zeros((bins, freq_bins))
        for j in range(0, bins):
            cond = bin_num == j
            mesh = np.ix_(cond, freq_list)
            correct_data = self.data[mesh]
            binned[j, :] = np.sum(correct_data, axis=0) / len(correct_data)
        binned[np.isnan(binned)] = 0
        return binned
