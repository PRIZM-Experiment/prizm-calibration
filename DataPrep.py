from data import Data

class DataPrep:
    
    def __init__(self, instrument, channel, year, calibration_type):
        self.instrument = instrument
        self.channel = channel
        self.year = year
        self.data = self.prep_data(instrument=self.instrument, channel=self.channel, year=self.year)
        self.antenna = self.get_antenna_data(self.data)
        self.shorts = self.get_shorts(self.data)
        self.lst = self.get_lst_time(self.data)
    
    def __call__(self, calibration_type)
        return self.get_data_product(calibration_type)
        
    @staticmethod
    def prep_data(instrument, channel, year):
    
        selections = {'100MHz':{'EW': {'2018': './selections/2018_100MHz_EW_Prototyping.p', 
                                   '2021': './selections/2021_100MHz_EW_Partial.p'}, 
                                'NS': {'2018': './selections/2018_100MHz_NS.p', 
                                   '2021':'./selections/2021_100MHz_NS_Partial.p'}
                               }, 
                      '70MHz':{'EW': {'2018': './selections/2018_70MHz_EW_Partial.p', 
                                  '2021': './selections/2021_70MHz_EW_Partial.p'}, 
                               'NS': {'2018':'./selections/2018_70MHz_NS_Partial.p', 
                                  '2021': './selections/2021_70MHz_NS_Partial.p'}
                              }
                     }
    
        data = Data.via_metadatabase(selection = selections[instrument][channel][year])
        
        data.partition(instruments=[instrument], channels=[channel], buffer=(1,1))
        
        data.lst(instruments=[instrument], channels=[channel])
        
        return data
    
    
    def get_antenna_data(self, data):
        
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='antenna')
    
    
    def get_shorts(self, data):
        
        return data.interpolate(times=data.get(data='time_sys_start', instrument=self.instrument, channel=self.channel, partition='antenna'), instrument=self.instrument, channel=self.channel, partition='short', threshold=5000)
    
    
    def get_lst_time(self, data):
        
        return data.get(data = 'lst_sys_start', instrument=instrument, channel=channel, partition='antenna')
    
    
    def prep_gsmcal_data(self):
        
        return self.antenna - self.shorts
    
    
    def get_data_product(self, calibration_type):
        
        calibrations = {'GSM': self.prep_gsmcal_data()}
        
        return calibrations[calibration_type]
        
        
        
  
