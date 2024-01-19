import sys
import numpy as np

mdb_default = '/scratch/s/sievers/nasuyu/prizm-data-wrangling/prizmdatawrangling'


class DataLoader():
    def __init__(self, freq, chan, data_path, mdb_path=mdb_default,):
        sys.path.append(mdb_path)
        import metadatabase as mdb
        from data import Data
        
        self.freq = freq
        self.chan = chan
        self.database = Data.via_metadatabase(selection=data_path)
        self.database.lst(instruments=[self.freq], channels=[self.chan])
        self.database.partition(instruments=[self.freq], channels=[self.chan], buffer=(1,1))
        
    def __call__(self, mid=True):
        data = self.get_data()
        lst = self.get_lst(mid=mid)
        systime = self.get_systime(mid=mid)
        return data, lst, systime
    
    def get_data(self):
        data = self.database.get(data='pol', instrument=self.freq, channel=self.chan, partition='antenna')
        return data
    
    def get_lst(self, mid=True):
        lst_start = self.database.get(data='lst_sys_start', instrument=self.freq, channel=self.chan, partition='antenna')
        lst_stop = self.database.get(data='lst_sys_stop', instrument=self.freq, channel=self.chan, partition='antenna')
        lst_mid = (lst_start + lst_stop) / 2
        if mid:
            return lst_mid
        return lst_start, lst_stop
    
    def get_systime(self, mid=True):
        systime_start = self.database.get(data='time_sys_start', instrument=self.freq, channel=self.chan, partition='antenna')
        systime_stop = self.database.get(data='time_sys_stop', instrument=self.freq, channel=self.chan, partition='antenna')
        systime_mid = (systime_start + systime_stop) / 2
        if mid:
            return systime_mid
        return systime_start, systime_stop
        