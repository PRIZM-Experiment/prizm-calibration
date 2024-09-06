import numpy as np
from scipy.signal import find_peaks


class Preprocess:
    def __init__(self, file_ending, path2file):
        with open(path2file + 'data' + file_ending, 'rb') as f:
            self.data = np.load(f)
        with open(path2file + 'lst' + file_ending, 'rb') as f:
            self.lst = np.load(f)
        # Should not be using systime
        with open(path2file + 'systime' + file_ending, 'rb') as f:
            self.systime = np.load(f)

    def __call__(self, crop=True, split_days=True):
        if crop:
            self.crop_data()
        if split_days:
            days = self.split_days()
        return self.data, self.lst, self.systime, days

    def crop_data(self):
        flow = f2i(30, flow=0, fhigh=250, num_inds=4096)
        fhigh = f2i(200, flow=0, fhigh=250, num_inds=4096)
        self.data = self.data[:, flow:fhigh]
        return self.data

    def split_days(self):
        days = find_peaks(self.lst)[0]
        days = np.insert(days, 0, 0)
        days = np.append(days, len(self.lst)-1)
        days += 1
        return days


def f2i(f, flow=30, fhigh=200, num_inds=2785):
    """convert freq to index"""
    return int((f-flow) * (num_inds/(fhigh-flow)))


def xrange(flow=30, fhigh=200, num_inds=2785):
    """xrange for plots"""
    return np.linspace(flow, fhigh, num_inds)


def lst2ind(t, binsize):
    binsize /= 60
    return int(t/binsize)
