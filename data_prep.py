import sys
from data import Data

class DataPrep:
    def __init__(self, instrument, channel, year):
        self.instrument = instrument
        self.channel = channel
        self.year = year
        self.data = self.prep_data()
        self.antenna = self.get_antenna_data(self.data)
        
        self.shorts = self.get_shorts(self.data)
        self.shorts_data = self.get_shorts_data(self.data)
        self.shorts_data_lst = self.get_shorts_lst_time(self.data)
        
        self.res50 = self.get_res50(self.data)
        self.res50_data = self.get_res50_data(self.data)
        self.res50_data_lst = self.get_res50_lst_time(self.data)
        
        self.res100 = self.get_res100(self.data)
        self.res100_data = self.get_res100_data(self.data)
        self.res100_data_lst = self.get_res100_lst_time(self.data)
        
        self.lst = self.get_lst_time(self.data)
        self.systime = self.get_sys_time(self.data)
    
    def __call__(self, calibration_type='GSM'):
        return self.get_data_product(calibration_type), self.lst, self.systime

    def prep_data(self):
        selections = {'100MHz':{'EW': {'2018': './selections/2018_100MHz_EW.p', 
                                   '2021': './selections/2021_100MHz_EW_Partial.p'}, 
                                'NS': {'2018': './selections/2018_100MHz_NS.p', 
                                   '2021': './selections/2021_100MHz_NS_Partial.p'}
                               }, 
                      '70MHz':{'EW': {'2018': './selections/2018_70MHz_EW_Partial.p', 
                                  '2021': './selections/2021_70MHz_EW_Partial.p'}, 
                               'NS': {'2018': './selections/2018_70MHz_NS_Partial.p', 
                                  '2021': './selections/2021_70MHz_NS_Partial.p'}
                              }
                     }
        data = Data.via_metadatabase(selection = selections[self.instrument][self.channel][self.year])
        data.partition(instruments=[self.instrument], channels=[self.channel], buffer=(1,1))
        data.lst(instruments=[self.instrument], channels=[self.channel])
        return data

    def get_antenna_data(self, data):
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='antenna')
    
    # -------------------- Short ------------------------------------ #
    def get_shorts(self, data):
        return data.interpolate(times=data.get(data='time_sys_start',
                                               instrument=self.instrument,
                                               channel=self.channel,
                                               partition='antenna'),
                                instrument=self.instrument,
                                channel=self.channel,
                                partition='short', threshold=5000)
    
    def get_shorts_lst_time(self, data):
        # Returns the LST times at which short calibrator spectra are taken
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='short')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='short')
        return (start + stop) / 2
    
    def get_shorts_data(self, data):
        # Saves only the measured spectra, *without* interpolating over antenna times
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='short')
    
    # --------------------- 50 Ohm ---------------------------------- #
    
    def get_res50(self,data):
        return data.interpolate(times=data.get(data='time_sys_start',
                                               instrument=self.instrument,
                                               channel=self.channel,
                                               partition='antenna'),
                                instrument=self.instrument,
                                channel=self.channel,
                                partition='res50', threshold=5000)
    
    def get_res50_lst_time(self, data):
        # Returns the LST times at which 50 Ohm calibrator spectra are taken
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='res50')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='res50')
        return (start + stop) / 2
    
    def get_res50_data(self, data):
        # Saves only the measured spectra, *without* interpolating over antenna times
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='res50')
    
    # ---------------------- 100 Ohm ------------------------------- #
    
    def get_res100(self,data):
        return data.interpolate(times=data.get(data='time_sys_start',
                                               instrument=self.instrument,
                                               channel=self.channel,
                                               partition='antenna'),
                                instrument=self.instrument,
                                channel=self.channel,
                                partition='res100', threshold=5000)
    
    def get_res100_lst_time(self, data):
        # Returns the LST times at which short calibrator spectra are taken
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='res100')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='res100')
        return (start + stop) / 2
    
    def get_res100_data(self, data):
        # Saves only the measured spectra, *without* interpolating over antenna times
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='res100')
    
    # ---------------------------- Antenna ------------------------- #
    

    def get_lst_time(self, data):
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='antenna')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='antenna')
        return (start + stop) / 2
    
    def get_sys_time(self, data):
        start = data.get(data='time_sys_start', instrument=self.instrument, channel=self.channel, partition='antenna')
        stop = data.get(data='time_sys_stop', instrument=self.instrument, channel=self.channel, partition='antenna')
        return (start + stop) / 2

    def prep_gsm_cal_data(self):
        # previously self.antenna - self.short, added in the flat 50 Ohm source calibration measurements
        return (self.antenna - self.shorts)/(self.res50 - self.shorts) 

    def get_data_product(self, calibration_type):
        if calibration_type == 'GSM':
            calibration_data = self.prep_gsm_cal_data()
        elif calibration_type == 'raw':
            calibration_data = self.antenna
        return calibration_data
    