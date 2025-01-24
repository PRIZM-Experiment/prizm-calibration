# A module to perform interpolation of PRIZM data sets using the Kriging (GPR) method

'''Import modules ------------------------------------------------'''
# Python modules
import numpy as np
import scipy
from matplotlib import pyplot as plt
# import copy
# import time

# Custom modules
# import data
# import data_prep
# try:
#     reload(data) # dependent module (imported in data_prep)
#     reload(data_prep)
#     from data_prep import DataPrep
# except:
#     from importlib import reload
#     reload(data) # dependent module (imported in data_prep)
#     reload(data_prep)
#     from data_prep import DataPrep

# from helper_functions import *
# import data_utils as du

# try:
#     reload(test_short_interp)
#     from test_short_interp import make_acf, make_acf_alt
# except:
#     import test_short_interp
#     reload(test_short_interp)
#     from test_short_interp import make_acf, make_acf_alt
'''---------------------------------------------------------------------'''

''' Frequency range ------------------------------------------'''
# Standard for PRIZM data
freqarr, freqstep = np.linspace(0,250,4096,retstep=True) # Based on number of frequency channels of the antenna
# minfreq = 70 # for testing
# maxfreq = 90 # for testing
# minfreqarg = int(minfreq/freqstep)
# maxfreqarg = int(maxfreq/freqstep)
'''-------------------------------------------------------------'''

'''Fitting functions -------------------------------------------'''
def linfunc(x,a,b):
    return a*x+b

def expfunc(x,a,b):
    return a*np.exp(-x/b)

def ACF_fit(a,b,functype='linear'):
    if functype == 'linear':
        return lambda dt: a*dt+b
    elif functype == 'exponential':
        return lambda dt: a*np.exp(-dt/b)
    
def delta_function(x, location=0, amplitude=0, width=1):
    delta = np.zeros_like(x)
    # Add a sharp peak at the specified location
    idx = np.argmin(np.abs(x - location))  # Find the index closest to the location
    if x[idx] == location:
        # we only want the peak at exactly zero
        delta[idx:idx+width] = amplitude  # Set the value to amplitude (height of delta peak)
    return delta
'''-------------------------------------------------------------'''


''' CLASS DEFINITION '''
class Kriging:
    def __init__(self,systime,data,interp_times,tmax_mask=np.nan,tmin_mask=np.nan,
                 minfreq=None,maxfreq=None,minfreqarg=None,maxfreqarg=None):
        '''
        Specify only one of these pairs of parameters: (minfreq,maxfreq) and (minfreqarg,maxfreqarg).
        
        Parameters:
        ------------
        systime: raw UNIX timestamp of the calibrator data timeseries, in seconds. Shape is 1D.
        data: raw calibrator ADC power at each systime and for each frequency channel. Shape is 2D: (time,frequency).
        interp_times: antenna times at which to interpolate the calibrator data, in seconds.
        tmax_mask: upper time limit for both the systime and the interp_times.
        tmin_mask: lower time limit for both the systime and the interp_times.
        minfreq: minimum frequency (in MHz) for which to compute interpolation.
        maxfreq: maximum frequency (in MHz) for which to compute interpolation.
        minfreqarg: minimum frequency channel (=index) for which to compute interpolation.
        maxfreqarg: maximum frequency channel (=index) for which to compute interpolation.
        '''
        self.data = data
        self.time = systime
        self.interp_times = interp_times
        
        if minfreq != None and maxfreq != None and minfreqarg == None and maxfreqarg == None:
            self.minfreq = minfreq
            self.minfreqarg = int(self.minfreq/freqstep)
            self.maxfreq = maxfreq
            self.maxfreqarg = int(self.maxfreq/freqstep)
            
        elif minfreq == None and maxfreq == None and minfreqarg != None and maxfreqarg != None:
            self.minfreqarg = minfreqarg
            self.minfreq = freqarr[self.minfreqarg]
            self.maxfreqarg = maxfreqarg
            self.maxfreq = freqarr[self.maxfreqarg]
        
        elif minfreq == None and maxfreq == None and minfreqarg == None and maxfreqarg == None:
            self.minfreqarg = 0
            self.minfreq = 0 # MHz
            self.maxfreqarg = 4095
            self.maxfreq = 250 # MHz
            
        else: raise ValueError('User must specify *either* min and max frequencies in MHz (minfreq, maxfreq), *or* min and max frequency channel number (minfreqarg, maxfreqarg). If neither is specified, the full frequency range is assumed.')
        
        # If tmin_mask and tmax_mask are specified then apply a mask (means we're only selecting a subset of the data)
        if (tmax_mask != np.nan) & (tmin_mask != np.nan):
            tmask=(self.time>tmin_mask)&(self.time<tmax_mask)
            self.time = self.time[tmask]
            self.data = self.data[tmask,:]
            tmask_ant=(self.interp_times>tmin_mask)&(self.interp_times<tmax_mask)
            self.interp_times = self.interp_times[tmask_ant]
    
    
    def __call__(self,dt=5,tmax=2*86400,dtmax=2*86400,acf_functype='quadpeak'):
        '''
        When called, the class computes the ACF, and computes the interpolation at every interp_time.
        
        Parameters
        -----------
        dt: timescale step for computation of the ACF. Default 5 seconds (<typical smallest timegap between measurements(6-7s)).
        tmax: Maximum timescale for calculation of ACF. In seconds. Default 2x86400 seconds (48 hours).
        dtmax: Maximum (cutoff) dt of the ACF used for Kriging. Default 2x86400 seconds (48 hours).
        acf_functype: 'linear', 'exponential', 'polyfit' (degree 4 polynomial), or 'quadpeak' (quadratic fit for dt>0, with added spike). Default quadpeak.
        '''
        
        print('Verifying freq args are the same within kriging object: min', self.minfreqarg, ', max', self.maxfreqarg)
        
        self.interp_data = np.zeros( shape=(len(freqarr[self.minfreqarg:self.maxfreqarg+1]),len(self.interp_times)) )
        self.interp_std = np.zeros( shape=(len(freqarr[self.minfreqarg:self.maxfreqarg+1]),len(self.interp_times)) )
        
        # We have to do the interpolation one frequency channel at a time
        for i,freq in enumerate(freqarr[self.minfreqarg:self.maxfreqarg+1]):
            #if i == 1: break # for testing, we break after 1 freq channel
            
            # Compute the ACF from data
            freq_index = int(freq/freqstep)
            tot,wt = self.make_acf(self.data[:,freq_index]-self.data[:,freq_index].mean(),self.time,dt,tmax)
            mm=wt>0.10*len(self.time) # mask ACF entries with less significant weights (<10% of number of datapoints) to get rid of noisy outliers
            self.acf_tvec = np.arange(len(tot))[mm]*dt
            self.acf = tot[mm]/wt[mm]
            
            # Smooth the ACF, overwrite ACF values/times with smoothed values
            self.acf_tvec, self.acf, self.acf_std = self.smooth_acf(tvec=self.acf_tvec,acf=self.acf,cycle_jump=5*60)
            
            # Fit the ACF from dt=0 to dt=dtmax
            self.acf_func = self.fit_acf(tvec=self.acf_tvec,acf=self.acf,acf_err=self.acf_std,dtmax=dtmax,functype=acf_functype)
            
            '''Here I change what data gets used for the weighted sum'''
            # Average the time series data taken within 1 calibration cycle before doing the weighted sum
            self.avgd_time, self.avgd_data, self.avgd_data_std = self.smooth_acf(self.time,self.data[:,freq_index],cycle_jump=5*60,isACF=False)
            # NOTE: the smooth_acf function can just be used as a general averaging function, so here ^ I'm using it to average the time series data (NOT the ACF)
            
            if i == 0:
                # First time around, initialized the matrices to save the averaged data.
                self.save_avgd_time = np.zeros( shape=(len(freqarr[self.minfreqarg:self.maxfreqarg+1]),
                                                       len(self.avgd_time)) )
                #self.save_avgd_time = self.avgd_time # normally, the time is the same for every freq channel
                self.save_avgd_data = np.zeros( shape=(len(freqarr[self.minfreqarg:self.maxfreqarg+1]),
                                                       len(self.avgd_data)) )
                self.save_avgd_data_std = np.zeros( shape=(len(freqarr[self.minfreqarg:self.maxfreqarg+1]),
                                                           len(self.avgd_data_std)) )

            # After the matrices are initialized, save current array to the matrix.
            self.save_avgd_time[i] = self.avgd_time
            self.save_avgd_data[i] = self.avgd_data
            self.save_avgd_data_std[i] = self.avgd_data_std
            
            # Perform Kriging for all antenna times for the current frequency channel
            for j, tt_interp in enumerate(self.interp_times):
                
                # Compute the covariance matrix for the current antenna time, and perform interpolation
                '''Using averaged time series data for the weighted sum'''
                self.CMatrix, self.interp_data[i,j],self.interp_std[i,j] = self.compute_covariance_and_krig(dat=self.avgd_data,t=self.avgd_time, dtmax=dtmax,interp_time=tt_interp,ACF_func=self.acf_func)
            
            print(freq,'MHz channel done')
    
        
    def make_acf(self,dat,t,dt,tmax):
        '''Function to compute the autocorrelation function of given data. Provided by Jon Sievers. Includes dt=0 bin.
        
        Parameters
        -----------
        dat: Time series data for which to compute the ACF. Must only include one frequency channel.
        t: Times correponding to each entry in dat. In seconds.
        dt: Time increment for calculation of the ACF. In seconds.
        tmax: Maximum timescale for calculation of ACF. In seconds.
        '''
        n=int(tmax/dt)
        tot=np.zeros(n)
        wt=np.zeros(n)
        nn=len(dat)
        for i in range(nn):
            # dt = 0 included if line below says range(i,nn) and dt<minimum-data-timestep. Not included if it says range(i+1,nn).
            for j in range(i,nn):
                delt=np.abs(t[i]-t[j])
                k=int(delt/dt) 
                if k<n:
                    tot[k]=tot[k]+dat[i]*dat[j]
                    wt[k]=wt[k]+1
                else:
                    break
        return tot,wt
    
    
    def smooth_acf(self,tvec,acf,cycle_jump=60,isACF=True):
        '''Function to smooth the ACF data by averaging ACF values obtained from calibrator data taken in a single calibrator measurement cycle, while conserving the dt=0 peak due to measurement noise. This translates to averaging data taken in close succession (~a few seconds time gap). Larger timegaps (e.g. order 0.5-1h) separate measurement cycles. We look for large peaks in the timegaps between subsequent data to separate into measurement cycles.
        
        Parameters
        -----------
        cycle_jump: Minimum timegape between subsequent measurements to separate them into two different measurement cycles, in seconds. Default 60 seconds.
        isACF: set to False if using this function for general averaging of another time of (non-ACF) dataset. Default True.
        
        
        '''
        # Separate the current dt values into bins within a few minutes of each other, I think this normally corresponds to data taken during the same rotation through calibrators before going back to antenna
        
        if isACF == True:
            time_values = tvec[1:] # we only smooth for dt>0
            acf_values = acf[1:] # "     "      "
            zerobin = acf[0] # save the zero-bin
        else:
            time_values = tvec
            acf_values = acf
            
        tsteps = np.diff(time_values)

        # Create bin edges based on big jumps in the timestep between subsequent measurements
        #bins = find_peaks(tsteps,height=cycle_jump)[0] 
        bins = np.where(tsteps>cycle_jump)[0] # bin edges, minimum jump for separate cycle set to 60 by default
        
        # Split data into the bins
        bin_tgroups = np.split(time_values,bins+1,axis=0)
        bin_acf_groups = np.split(acf_values,bins+1,axis=0)
        
        t_binavg = []
        acf_binavg = []
        acf_binstd = []
        #largest_binwidth = 0 # for testing

        for i in range(len(bin_tgroups)):
            t_binavg.append(np.mean(bin_tgroups[i]))
            acf_binavg.append(np.mean(bin_acf_groups[i]))
            acf_binstd.append(np.std(bin_acf_groups[i]))
            # Code below is for testing
#             if (bin_tgroups[i][-1]-bin_tgroups[i][0]) > 5*60:
#                 print('bin larger than 5 minutes:',bin_tgroups[i][-1]-bin_tgroups[i][0])
#             if (bin_tgroups[i][-1]-bin_tgroups[i][0] > largest_binwidth):
#                 largest_binwidth = bin_tgroups[i][-1]-bin_tgroups[i][0]
#                 largest_bini = i

#         print(largest_binwidth/60,'mins')
#         print(largest_bini)

        # Add the dt=0 datapoint back in post-smoothing
        if isACF == True:
            smoothed_times = np.concatenate(([0],t_binavg))
            smoothed_acf = np.concatenate(([zerobin],acf_binavg))
            smoothed_std = np.concatenate(([0],acf_binstd)) # here we assume the dt=0 value is exact
        else:
            smoothed_times = np.array(t_binavg)
            smoothed_acf = np.array(acf_binavg)
            smoothed_std = np.array(acf_binstd)
        
        return smoothed_times, smoothed_acf, smoothed_std
    
    
    def fit_acf(self,tvec,acf,acf_err,dtmax=2*86400,functype='quadpeak'):
        '''Function to fit the given ACF to either a linear or exponential model. These simplistic models are only meant
        to approximate the early part of the ACF (<dtmax), and should not be used to model the full ACF.
        
        Parameters
        -----------
        tvec: array of timescales dt at which the ACF is computed. In seconds.
        acf: value of the ACF at each dt in tvec.
        acf_err: error on the ACF at each dt in tvec.
        dtmax: Maximum (cutoff) dt of the ACF used for Kriging. Default 2x86400 seconds (48 hours).
        functype: 'linear', 'exponential', 'polyfit' (degree 4 polynomial), or 'quadpeak' (quadratic fit for dt>0, with added spike).
        of the height of ACF at dt=0). Default 'quadpeak'. Defines what model will be used to fit the ACF.
        '''
        if functype == 'quadpeak':
            # Quadratic for 0<dt<dtmax + dt=0 peak height
            c, cov = np.polyfit(x=tvec[(0<tvec)&(tvec<dtmax)]/3600,y=acf[(0<tvec)&(tvec<dtmax)],w=1/acf_err[(0<tvec)&(tvec<dtmax)],deg=2,cov=True)
            c_errs = np.diag(cov)**(1/2) # errors on the fit coefficients, for now not used for anything
            polyfunc = np.poly1d(c)

            zeropeak_height = abs(acf[0] - acf[1])

            def ACF_func(dt):
                return polyfunc(dt) + delta_function(dt,amplitude=zeropeak_height)
        
        elif functype == 'linear':
            fit_func = linfunc
            popt, pcov = scipy.optimize.curve_fit(f=fit_func,xdata=tvec[tvec<dtmax]/3600,ydata=acf[tvec<dtmax],p0=[-6e10/8,9.1e10])
            ACF_func = ACF_fit(popt[0],popt[1],functype=functype)
        elif functype == 'exponential':
            fit_func = expfunc
            popt, pcov = scipy.optimize.curve_fit(f=fit_func,xdata=tvec[tvec<dtmax]/3600,ydata=acf[tvec<dtmax],p0=[9.1e10,4])
            ACF_func = ACF_fit(popt[0],popt[1],functype=functype)
        elif functype == 'polyfit':
            c = np.polyfit(x=tvec[tvec<dtmax]/3600,y=acf[tvec<dtmax],deg=4)
            ACF_func = np.poly1d(c)
        
        return ACF_func
    
    
    def compute_covariance_and_krig(self,dat,t,dtmax,interp_time,ACF_func):
        '''Function to compute the covariance matrix and perform steps for Kriging.
        
        Parameters
        -----------
        dat: raw ADC power data to interpolate.
        t: raw UNIX timestamp time for each dat, in seconds.
        dtmax: Maximum (cutoff) dt of the ACF used for Kriging. Default 2x86400 seconds (48 hours).
        interp_time: antenna times at which to interpolate dat, in seconds.
        ACF_func: function modelling the ACF of dat, fit for at least 0<=dt<=dtmax. Takes dt input in **hours**.
        '''
        
        # Making a matrix with only the data within dtmax/2 of the interp time, to insure we only use up to dtmax of the ACF.
        d_red = dat[abs(t - interp_time) < dtmax/2] 
        t_red = t[abs(t - interp_time) < dtmax/2]
        
        # this line skips interp_times that are not within dtmax/2 of the measured data
        if len(t_red) == 0: return np.nan, np.nan, np.nan 
        
        d = np.append(d_red,0)
        tarr = np.append(t_red,interp_time)
        C = np.zeros(shape=(len(d),len(d))) # initialize the covariance matrix
                    
        for i in range(len(d)):
            ti = tarr[i]
            dtij_arr = abs(ti-tarr) # full array of the time separation from ti
            #if np.any(dtij_arr>dtmax): print('larger') # normally this should not happen since we've already truncated the array

            C[i,:] = ACF_func(dtij_arr/3600) # ACF_func is defined for dt in hours
         
        C += 1e20
        
        # Invert the covariance matrix
        Cinv = np.linalg.inv(C)
        
        # "w": array of weights for the weighted sum to compute inteprolated value
        n = len(d)-1
        w = -Cinv[n,0:n] / Cinv[n,n]
        
        # compute weighted sum to find interpolated value
        dinterp = np.dot(w,d[0:n])
        
        # compute error on interpolated value
        interp_std = np.sqrt(1/Cinv[n,n]) # standard deviation
        
        return C, dinterp, interp_std
