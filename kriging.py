# A module to perform interpolation of PRIZM data sets using the Kriging (GPR) method

'''Import modules ------------------------------------------------'''
# Python modules
import numpy as np
import scipy
from matplotlib import pyplot as plt
import copy

# Custom modules
import data
import data_prep
try:
    reload(data) # dependent module (imported in data_prep)
    reload(data_prep)
    from data_prep import DataPrep
except:
    from importlib import reload
    reload(data) # dependent module (imported in data_prep)
    reload(data_prep)
    from data_prep import DataPrep

from helper_functions import *
import data_utils as du

try:
    reload(test_short_interp)
    from test_short_interp import make_acf, make_acf_alt
except:
    import test_short_interp
    reload(test_short_interp)
    from test_short_interp import make_acf, make_acf_alt
'''---------------------------------------------------------------------'''

''' Frequency range ------------------------------------------'''
freqarr, freqstep = np.linspace(0,250,4096,retstep=True) # Based on number of frequency channels of the antenna
minfreq = 70
maxfreq = 90
minfreqarg = int(minfreq/freqstep)
maxfreqarg = int(maxfreq/freqstep)
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
'''-------------------------------------------------------------'''




''' CLASS DEFINITION '''
class Kriging:
    def __init__(self,systime,data,interp_times,tmax_mask=np.nan,tmin_mask=np.nan):
        self.data = data
        self.time = systime
        self.interp_times = interp_times
        
        # If tmin_mask and tmax_mask are specified then apply a mask (means we're only selecting a subset of the data)
        if (tmax_mask != np.nan) & (tmin_mask != np.nan):
            tmask=(self.time>tmin_mask)&(self.time<tmax_mask)
            self.time = self.time[tmask]
            self.data = self.data[tmask,:]
            tmask_ant=(self.interp_times>tmin_mask)&(self.interp_times<tmax_mask)
            self.interp_times = self.interp_times[tmask_ant]
    
    
    def __call__(self,dt,tmax=2*86400,dtmax=2*86400,acf_functype='linear'):
        '''
        When called, the class computes the ACF, and computes the interpolation at every interp_time.
        
        Parameters
        -----------
        dt: timescale step for computation of the ACF.
        tmax: Maximum timescale for calculation of ACF. In seconds. Default 2x86400 seconds (48 hours).
        dtmax: Maximum (cutoff) dt of the ACF used for Kriging. Default 2x86400 seconds (48 hours).
        acf_functype: 'linear' or 'exponential', default 'linear'. Defines what model will be used to fit the ACF.
        '''
        self.interp_data = np.zeros( shape=(len(freqarr[minfreqarg:maxfreqarg]),len(self.interp_times)) )
        self.interp_std = np.zeros( shape=(len(freqarr[minfreqarg:maxfreqarg]),len(self.interp_times)) )
        
        # We have to do the interpolation one frequency channel at a time
        for i,freq in enumerate(freqarr[minfreqarg:maxfreqarg]):
            if i == 1: break # for testing
            # Compute the ACF from data
            freq_index = int(freq/freqstep)
            tot,wt = self.make_acf(self.data[:,freq_index]-self.data[:,freq_index].mean(),self.time,dt,tmax)
            mm=wt>30 # mask ACF entries with insignificant weights?
            self.acf_tvec = np.arange(len(tot))[mm]*dt
            self.acf = tot[mm]/wt[mm]
            
            # Fit the ACF from dt=0 to dt=dtmax
            self.acf_func = self.fit_acf(tvec=self.acf_tvec,acf=self.acf,dtmax=dtmax,functype=acf_functype)
            
            # Perform Kriging for all antenna times for the current frequency channel
            for j, tt_interp in enumerate(self.interp_times):
                
                # Compute the covariance matrix for the current antenna time, and perform interpolation
                self.CMatrix, self.interp_data[i,j],self.interp_std[i,j] = self.compute_covariance_and_krig(dat=self.data[:,freq_index],t=self.time,
                                                       dtmax=dtmax,interp_time=tt_interp,ACF_func=self.acf_func)
                
    
        
    def make_acf(self,dat,t,dt,tmax):
        '''Function to compute the autocorrelation function of given data. Provided by Jon Sievers.
        
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
    
    
    def fit_acf(self,tvec,acf,dtmax,functype='linear'):
        '''Function to fit the given ACF to either a linear or exponential model. These simplistic models are only meant
        to approximate the early part of the ACF (<dtmax), and should not be used to model the full ACF.
        
        Parameters
        -----------
        tvec: array of timescales dt at which the ACF is computed.
        acf: value of the ACF at each dt in tvec.
        dtmax: Maximum (cutoff) dt of the ACF used for Kriging. Default 2x86400 seconds (48 hours).
        functype: 'linear' or 'exponential', default 'linear'. Defines what model will be used to fit the ACF.
        '''
        
        if functype == 'linear':
            fit_func = linfunc
        elif functype == 'exponenetial':
            fit_func = expfunc
        
        popt, pcov = scipy.optimize.curve_fit(f=fit_func,xdata=tvec[tvec<dtmax]/3600,ydata=acf[tvec<dtmax],p0=[-6e10/8,9.1e10])
        
        ACF_func = ACF_fit(popt[0],popt[1],functype=functype)
        
        return ACF_func
    
    
    def compute_covariance_and_krig(self,dat,t,dtmax,interp_time,ACF_func):
        '''Function to compute the covariance matrix and perform steps for Kriging.
        
        Parameters
        -----------
        dat:
        t:
        dtmax:
        interp_time:
        ACF_func:
        '''
        
        # Making a matrix with only the data within dtmax/2 of the interp time, to insure we only use up to dtmax of the ACF.
        d_red = dat[abs(t - interp_time) < dtmax/2] 
        t_red = t[abs(t - interp_time) < dtmax/2]
        d = np.append(d_red,0)
        tarr = np.append(t_red,interp_time)
        C = np.zeros(shape=(len(d),len(d))) # initialize the covariance matrix
        
        for i in range(len(d)):
            for j in range(len(d)):
                dtij = abs(tarr[i]-tarr[j]) # time separation between points

                if dtij > dtmax:
                    # For now, we are only using the ACF up to dt = dtmax. If dtij > dtmax, set C_ij = 0 (uncorrelated).
                    # Normally this should not happen in the reduced matrix format we set up above.
                    C[i,j] = 0
                else:
                    C[i,j] = ACF_func(dt=dtij/3600) # ACF_func is defined for dt in hours
         
        
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