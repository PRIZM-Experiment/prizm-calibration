# Sample .py file for executing Kriging on time series data

import time

tstart = time.time()
import sys
import numpy as np
import matplotlib.pyplot as plt
tend = time.time()
print('Time taken for loading Python modules:',(tend-tstart)/60,' min')

# Collect command line arguments
if __name__ == "__main__":
	minfreqarg = int(sys.argv[1])
	maxfreqarg = min(4095,int(sys.argv[2])) # the maximum allowed freq index is 4095
	print('Got freq limits:',minfreqarg,maxfreqarg)

tstart = time.time()
from kriging import Kriging
tend = time.time()
print('Time taken for loading in kriging.py:',(tend-tstart)/60,' min')

# Define the frequency array for the complete dataset
freqarr, freqstep = np.linspace(0,250,4096,retstep=True) # Based on number of frequency channels of the antenna

# Load in the data
year = '2021'
instrument = '100MHz'
channel = 'NS'

datadir = '../test_data/'+year+'/'+instrument+'/'+channel+'/'

tstart = time.time()
save_mask = np.load(datadir+'shortdata_'+year+'_'+instrument+'_'+channel+'_mask.npy')
dat = np.load(datadir+'shortdata_meas_'+year+'_'+instrument+'_'+channel+'.npy')[save_mask]
lst = np.load(datadir+'shortlst_'+year+'_'+instrument+'_'+channel+'.npy')[save_mask]
t = np.load(datadir+'shortsystime_'+year+'_'+instrument+'_'+channel+'.npy')[save_mask]
t_ant = np.load(datadir+'antsystime_'+year+'_'+instrument+'_'+channel+'.npy')
tend = time.time()

print('Time taken for loading in data:',(tend-tstart)/60,' min')

# Set some Kriging parameters
tmax=np.inf
tmin=0

# Initialize Kriging class
tstart = time.time()
krig_object = Kriging(systime=t,data=dat,interp_times=t_ant,tmax_mask=tmax,tmin_mask=tmin,minfreqarg=minfreqarg,maxfreqarg=maxfreqarg)
tend = time.time()
print('Time taken for initializing Kriging class:',(tend-tstart)/60,' min')

# Call object to perform Kriging
tstart = time.time()
krig_object(dt=5,dtmax=10*3600,acf_functype='quadpeak') # dt and dtmax in seconds
tend = time.time()
print('Time taken for Kriging:',(tend-tstart)/60,' min')


# Collect results, all freqs
rslt = krig_object.interp_data
err = krig_object.interp_std
avgd_time = krig_object.save_avgd_time
avgd_data = krig_object.save_avgd_data
avgd_data_std = krig_object.save_avgd_data_std

# Save results
tstart = time.time()
save_dir = '/scratch/s/sievers/lauriea/2021-100NS-short/'+str(minfreqarg)+'_'+str(maxfreqarg)+'/'
save_path = save_dir + 'short_'+str(minfreqarg)+'_'+str(maxfreqarg)
np.save(save_path+'_rslt.npy',rslt)
np.save(save_path+'_rslterr.npy',err)
np.save(save_path+'_avgtime.npy',avgd_time)
np.save(save_path+'_avgdata.npy',avgd_data)
np.save(save_path+'_avgdatastd.npy',avgd_data_std)
tend = time.time()
print('Time taken for saving results data:',(tend-tstart)/60,' min')

# Plot results
#freq = 70 # MHz
#freq_index = int(freq/freqstep)

#fig = plt.figure(figsize=(20,5))
#plt.rcParams.update({'font.size': 12})

#plt.errorbar(x=avgd_time,y=avgd_data,yerr=avgd_data_std,marker='.',label='Real data',ls='none',zorder=1)
#plt.plot(tt_ant,rslt,'.',ms=1,label='Interp Data',color='k',zorder=2)
#plt.fill_between(tt_ant,(rslt-err),(rslt+err),alpha=0.5,color='k',zorder=0,label=r'1$\sigma$ on Interp Data')

#plt.xlabel('UNIX Timestamp [s]')
#plt.ylabel('ADC Power')
#plt.legend()
#plt.title('Interpolation Results for a Subset of '+year+' '+instrument+' '+channel+' Data\nFrequency Channel = '+str(freq)+'MHz')
#plt.plot(tt_ant,[6.6e7 for ti in tt_ant],'.',color='red',label='Antenna times (interp times)')

#plt.savefig('/scratch/s/sievers/lauriea/short-job-test.png',format='png',dpi=150)
#plt.show()
