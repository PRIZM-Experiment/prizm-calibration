import sys
from data import Data # note the 'data' variable in this file is NOT the data module (which is not explicitly imported), it is an instance of the Data class
import numpy as np

# For now, this is only used for calibrator spectra flagging ---------------
freqarr, freqstep = np.linspace(0,250,4096,retstep=True) # Based on number of frequency channels of the antenna
minfreq = 30
maxfreq = 200
minfreqarg = int(minfreq/freqstep)
maxfreqarg = int(maxfreq/freqstep)
# -----------------------------------------

class DataPrep:
    def __init__(self, instrument, channel, year):
        self.instrument = instrument
        self.channel = channel
        self.year = year
        self.data = self.prep_data()
        
        # Get raw antenna spectra
        self.antenna = self.get_antenna_data(self.data)
        self.lst = self.get_lst_time(self.data)
        self.systime = self.get_sys_time(self.data)
        
        # Get the raw calibrator spectra
        self.shorts_data = self.get_shorts_data(self.data)
        self.shorts_data_lst = self.get_shorts_lst_time(self.data)
        self.shorts_data_systime = self.get_shorts_sys_time(self.data)
        
        self.res50_data = self.get_res50_data(self.data)
        self.res50_data_lst = self.get_res50_lst_time(self.data)
        self.res50_data_systime = self.get_res50_sys_time(self.data)
        
        self.res100_data = self.get_res100_data(self.data)
        self.res100_data_lst = self.get_res100_lst_time(self.data)
        self.res100_data_systime = self.get_res100_sys_time(self.data)
        
        '''UNDER DEVELOPMENT'''
        # Generate a mask of bad/outlier calibrator spectra
        self.calib_mask_dict = {'short':self.flag_calibrator_data(self.shorts_data,'short'),
                                'res50':self.flag_calibrator_data(self.res50_data,'res50'),
                                'res100':self.flag_calibrator_data(self.res100_data,'res100')}
        
        # Flag bad spectra, then do interpolation
        # self.calib_mask_dict is retrieved within get_shorts (or res50, res100 equiv.)
        self.shorts = self.get_shorts(self.data)
        self.res50 = self.get_res50(self.data)
        self.res100 = self.get_res100(self.data)
    
    def __call__(self, calibration_type='GSM'):
        return self.get_data_product(calibration_type), self.lst, self.systime

    def prep_data(self):
        selections = {'100MHz':{'EW': {'2018': './selections/2018_100MHz_EW_Prototyping.p', 
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
        
        # Load in the full dataset from pickle file
        data = Data.via_metadatabase(selection = selections[self.instrument][self.channel][self.year])
        
        # Partition the dataset
        # Special case: 2021 100MHz EW/NS are swapped in the .p file.
        if self.year == '2021' and self.instrument == '100MHz':
            # Switch the channel attribute
            if self.channel == 'EW': self.channel = 'NS'
            else: self.channel = 'EW'
        
        data.partition(instruments=[self.instrument], channels=[self.channel], buffer=(1,1))
        data.lst(instruments=[self.instrument], channels=[self.channel])
        return data

    def get_antenna_data(self, data):
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='antenna')
    
    def get_lst_time(self, data):
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='antenna')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='antenna')
        return (start + stop) / 2
    
    def get_sys_time(self, data):
        start = data.get(data='time_sys_start', instrument=self.instrument, channel=self.channel, partition='antenna')
        stop = data.get(data='time_sys_stop', instrument=self.instrument, channel=self.channel, partition='antenna')
        return (start + stop) / 2
    
    
    # -------------------- Short ------------------------------------ #
    def get_shorts(self, data):
        return data.interpolate(times=data.get(data='time_sys_start',
                                               instrument=self.instrument,
                                               channel=self.channel,
                                               partition='antenna'),
                                calib_mask=self.calib_mask_dict['short'],
                                instrument=self.instrument,
                                channel=self.channel,
                                partition='short', threshold=3600*24) # set to 24h for testing purposes
    
    def get_shorts_lst_time(self, data):
        # Returns the LST times at which short calibrator spectra are taken
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='short')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='short')
        return (start + stop) / 2
    
    def get_shorts_sys_time(self, data):
        # Returns the LST times at which short calibrator spectra are taken
        start = data.get(data='time_sys_start', instrument=self.instrument, channel=self.channel, partition='short')
        stop = data.get(data='time_sys_stop', instrument=self.instrument, channel=self.channel, partition='short')
        return (start + stop) / 2
    
    def get_shorts_data(self, data):
        # Saves only the measured spectra, *without* interpolating over antenna times
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='short')
    
    # --------------------- 50 Ohm ---------------------------------- #
    
    def get_res50(self, data):
        return data.interpolate(times=data.get(data='time_sys_start',
                                               instrument=self.instrument,
                                               channel=self.channel,
                                               partition='antenna'),
                                calib_mask=self.calib_mask_dict['res50'],
                                instrument=self.instrument,
                                channel=self.channel,
                                partition='res50', threshold=5000)
    
    def get_res50_lst_time(self, data):
        # Returns the LST times at which 50 Ohm calibrator spectra are taken
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='res50')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='res50')
        return (start + stop) / 2
    
    def get_res50_sys_time(self, data):
        # Returns the LST times at which 50 Ohm calibrator spectra are taken
        start = data.get(data='time_sys_start', instrument=self.instrument, channel=self.channel, partition='res50')
        stop = data.get(data='time_sys_stop', instrument=self.instrument, channel=self.channel, partition='res50')
        return (start + stop) / 2
    
    def get_res50_data(self, data):
        # Saves only the measured spectra, *without* interpolating over antenna times
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='res50')
    
    # ---------------------- 100 Ohm ------------------------------- #
    
    def get_res100(self, data):
        return data.interpolate(times=data.get(data='time_sys_start',
                                               instrument=self.instrument,
                                               channel=self.channel,
                                               partition='antenna'),
                                calib_mask=self.calib_mask_dict['res100'],
                                instrument=self.instrument,
                                channel=self.channel,
                                partition='res100', threshold=5000)
    
    def get_res100_lst_time(self, data):
        # Returns the LST times at which 100 Ohm calibrator spectra are taken
        start = data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='res100')
        stop = data.get(data='lst_sys_stop', instrument=self.instrument, channel=self.channel, partition='res100')
        return (start + stop) / 2
    
    def get_res100_sys_time(self, data):
        # Returns the LST times at which 100 Ohm calibrator spectra are taken
        start = data.get(data='time_sys_start', instrument=self.instrument, channel=self.channel, partition='res100')
        stop = data.get(data='time_sys_stop', instrument=self.instrument, channel=self.channel, partition='res100')
        return (start + stop) / 2
    
    def get_res100_data(self, data):
        # Saves only the measured spectra, *without* interpolating over antenna times
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='res100')
    
    # ---------------------------- Calibration Functions ------------------------- #
    
    def flag_calibrator_data(self,calib_data,partition):
        '''
        Parameters
        ----------
        calib_data: contains array of calibrator data of dimension (number of spectra) X (number of frequency channels)
        partition: calibrator being flagged; 'short', 'res50' or 'res100'.
        '''
        
        # intialize calib mask to all 'True', aka no masking
        calib_mask = np.full(shape=calib_data.shape[0],fill_value=True)
        
        # set thresholds (on the median spectrum over all data from this partition)
        flag_thresholds = {'100MHz':{'EW': 
                                         {'2018': {'short':{'index': maxfreqarg, 'threshold': 0.05}},
                                        '2021': {'short':{'index': maxfreqarg, 'threshold': 0.15}}}, 
                                    'NS': {'2018': {'short':{'index': maxfreqarg, 'threshold': 0.09}}, 
                                           '2021': {'short':{'index': maxfreqarg, 'threshold': 0.03}}}
                                }, 
                          '70MHz':{'EW': 
                                       {'2018': {'short':{'index': maxfreqarg, 'threshold': 0.03}}, 
                                      '2021': {'short':{'index': minfreqarg, 'threshold': 0.15}}}, 
                                   'NS': 
                                        {'2018': {'short':{'index': minfreqarg, 'threshold': 0.015},
                                                 'res50':{'index': minfreqarg, 'threshold': 0.019},
                                                 'res100':{'index': minfreqarg, 'threshold': 0.022}}}
                                }
                          }
        
        # thresholds are set on residuals from the time median spectrum over all data, so compute residuals
        time_median = np.median(calib_data,axis=0)
        res_calib_data = np.abs((calib_data - time_median)/time_median)
        
        # select correct thresholds
        # special case: 2021 100MHz EW/NS are swapped in the .p file. We switched self.channel, but for the 
        # flag_threshold dictionary key we need the real polarization (the one that was initially inputted by the user.)
        if self.year == '2021' and self.instrument == '100MHz':
            # Switch the channel attribute
            if self.channel == 'EW': channel_key = 'NS'
            else: channel_key = 'EW'
        else: channel_key = self.channel
        
        try:
            thresh_selection = flag_thresholds[self.instrument][channel_key][self.year][partition]
            idx_threshold, threshold = thresh_selection['index'], thresh_selection['threshold']
        except:
            print('Returning default mask')
            return calib_mask # still all 'True', so no masking
        
        # generate the mask based on residuals from time median computed above.
        # based on how the thresholds were selected, we want the mask to say 'False' when above the threshold.
        calib_mask = res_calib_data[:,idx_threshold] <= threshold 
        
        return calib_mask
    
    def get_flagged_calibrator_data(self,partition,return_flagged=False):
        '''Returns 'clean' calibrator data (LST and post-flagging spectra). For data inspection and cleaning purposes.
        
        Parameters
        ----------
        partition: which cleaned calibrator data to return; 'short', 'res50', or 'res100'.
        return_flagged: set to 'True' to return only the bad/outlier calibrator data.
        '''
        calib_list = {'short': [self.shorts_data_lst,self.shorts_data],
                     'res50': [self.res50_data_lst,self.res50_data],
                     'res100': [self.res100_data_lst,self.res100_data]}
        calib_data = calib_list[partition]
        calib_mask = self.calib_mask_dict[partition]
        
        # Apply flag
        if return_flagged == True:
            flagged_data = [calib_data[0][~calib_mask],calib_data[1][~calib_mask]]
        else: 
            flagged_data = [calib_data[0][calib_mask],calib_data[1][calib_mask]]
        
        return flagged_data

    def prep_gsm_cal_data(self):
        # previously self.antenna - self.short, added in the flat 50 Ohm source calibration measurements
        return (self.antenna - self.shorts)/(self.res50 - self.shorts) 

    def get_data_product(self, calibration_type):
        if calibration_type == 'short':
            calibration_data = self.antenna - self.shorts
        elif calibration_type == 'res50':
            calibration_data = (self.antenna - self.shorts)/(self.res50 - self.shorts)
        elif calibration_type == 'res100':
            calibration_data = (self.antenna - self.shorts)/(self.res100 - self.shorts)
        elif calibration_type == 'raw':
            calibration_data = self.antenna
        return calibration_data
    
