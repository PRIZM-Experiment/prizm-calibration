from data import Data


class DataPrep:

    def __init__(self, instrument, channel, year):
        self.instrument = instrument
        self.channel = channel
        self.year = year
        self.data = self.prep_data()
        self.antenna = self.get_antenna_data(self.data)
        self.shorts = self.get_shorts(self.data)
        self.lst = self.get_lst_time(self.data)
    
    def __call__(self, calibration_type):
        return self.get_data_product(calibration_type), self.lst

    def prep_data(self):
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
        data = Data.via_metadatabase(selection = selections[self.instrument][self.channel][self.year])
        data.partition(instruments=[self.instrument], channels=[self.channel], buffer=(1,1))
        data.lst(instruments=[self.instrument], channels=[self.channel])
        return data

    def get_antenna_data(self, data):
        return data.get(data='pol', instrument=self.instrument, channel=self.channel, partition='antenna')

    def get_shorts(self, data):
        return data.interpolate(times=data.get(data='time_sys_start',
                                               instrument=self.instrument,
                                               channel=self.channel,
                                               partition='antenna'),
                                instrument=self.instrument,
                                channel=self.channel,
                                partition='short', threshold=5000)

    def get_lst_time(self, data):
        return data.get(data='lst_sys_start', instrument=self.instrument, channel=self.channel, partition='antenna')

    def prep_gsm_cal_data(self):
        return self.antenna - self.shorts

    def get_data_product(self, calibration_type):
        if calibration_type == 'GSM':
            calibration_data = self.prep_gsm_cal_data()
        else:
            calibration_data = None
        return calibration_data

