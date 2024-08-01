# Collection of utility functions for dealing with PRIZM antenna and calibrator data, for purposes such as
# - saving/loading data from local files
# - generating waterfall plots of the spectra
# - generating waterfall plots of the variation of spectra from the median spectrum over time
# - splitting the data into 24h LST days
# Written by Laurie Amen (June 19, 2024)

# Import statements:
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from helper_functions import *
# from data_prep import DataPrep

freqarr_default, freqstep = np.linspace(0,250,4096,retstep=True)

# ---------------------------------------------------------------------------------------------------- #
def save_all_data(filepath,dp,instrument,channel,year,SaveAnt=False,SaveInterpCalib=False):
    '''
    Function to save all data from dataprep. Utility to avoid reloading from metadatabase every time. Saves only unsplit data stored in DataPrep instances.
    Future additions: add to the plotting function as an optional call?
    
    Parameters
    ------------
    dp: DataPrep instance containing the data
    SaveAnt: Indicates if antenna data should be saved. Default is False (antenna data is NOT saved).
    SaveInterpCalib: Indicates if calibrator data interpolated over antenna times should be saved. Default is False (interpolated calibrator data is NOT saved).
    
    '''
    
    file_end = '_'+year+'_'+instrument+'_'+channel+'.npy'
    
    if SaveAnt == False and SaveInterpCalib == False:
        # DEFAULT
        # Save: everything, EXCEPT antenna and EXCEPT interpolated calibrators
        arrays = [dp.lst, dp.systime,
                     dp.shorts_data, dp.shorts_data_lst, dp.shorts_data_systime,
                     dp.res50_data, dp.res50_data_lst, dp.res50_data_systime,
                     dp.res100_data, dp.res100_data_lst, dp.res100_data_systime]
        names = ['antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime',
                    'res50data_meas','res50lst','res50systime',
                    'res100data_meas','res100lst','res100systime']
        
    
    elif SaveAnt == True and SaveInterpCalib == False:
        # Save: antenna (and everything else EXCEPT interpolated calibrators)
        arrays = [dp.antenna, dp.lst, dp.systime,
                     dp.shorts_data, dp.shorts_data_lst, dp.shorts_data_systime,
                     dp.res50_data, dp.res50_data_lst, dp.res50_data_systime,
                     dp.res100_data, dp.res100_data_lst, dp.res100_data_systime]
        names = ['antdata', 'antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime',
                    'res50data_meas','res50lst','res50systime',
                    'res100data_meas','res100lst','res100systime']
        
    elif SaveAnt == False and SaveInterpCalib == True:
        # Save: interpolated calibrators (and everything else EXCEPT antenna)
        arrays = [dp.lst, dp.systime,
                     dp.shorts_data, dp.shorts_data_lst, dp.shorts_data_systime, dp.shorts,
                     dp.res50_data, dp.res50_data_lst, dp.res50_data_systime, dp.res50,
                     dp.res100_data, dp.res100_data_lst, dp.res100_data_systime, dp.res100]
        names = ['antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime','shortinterp',
                    'res50data_meas','res50lst','res50systime','res50interp',
                    'res100data_meas','res100lst','res100systime','res100interp']
        
    else:
        # Implies Save: antenna and interpolated calibrators (and everything else)
        arrays = [dp.antenna, dp.lst, dp.systime,
                     dp.shorts_data, dp.shorts_data_lst, dp.shorts_data_systime, dp.shorts,
                     dp.res50_data, dp.res50_data_lst, dp.res50_data_systime, dp.res50,
                     dp.res100_data, dp.res100_data_lst, dp.res100_data_systime, dp.res100]
        names = ['antdata', 'antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime','shortinterp',
                    'res50data_meas','res50lst','res50systime','res50interp',
                    'res100data_meas','res100lst','res100systime','res100interp']

    for i in range(len(names)):
        np.save(filepath + names[i] + file_end, arrays[i])
    
    return


# -------------------------------------------------------------------------------------------------- #
def load_all_data(filepath,instrument,channel,year,LoadAnt=False,LoadInterpCalib=False):
    '''
    Function to load in antenna and calibrator data that's previously been saved.
    
    Parameters
    ----------
    LoadAnt: Indicates if antenna data should be loaded in. Default is False (antenna data is NOT loaded in).
    LoadInterpCalib: Indicates if calibrator data interpolated over antenna times should be loaded in. Default is False (interpolated calibrator data is NOT loaded in). 
    
    '''
    
    file_end = '_'+year+'_'+instrument+'_'+channel+'.npy'
    
    if LoadAnt == False and LoadInterpCalib == False:
        # DEFAULT
        # Load: everything, EXCEPT antenna and EXCEPT interpolated calibrators
        names = ['antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime',
                    'res50data_meas','res50lst','res50systime',
                    'res100data_meas','res100lst','res100systime']
        
    
    elif LoadAnt == True and LoadInterpCalib == False:
        # Load: antenna (and everything else EXCEPT interpolated calibrators)
        names = ['antdata', 'antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime',
                    'res50data_meas','res50lst','res50systime',
                    'res100data_meas','res100lst','res100systime']
        
    elif LoadAnt == False and LoadInterpCalib == True:
        # Load: interpolated calibrators (and everything else EXCEPT antenna)
        names = ['antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime','shortinterp',
                    'res50data_meas','res50lst','res50systime','res50interp',
                    'res100data_meas','res100lst','res100systime','res100interp']
        
    else:
        # Implies Load: antenna and interpolated calibrators (and everything else)
        names = ['antdata', 'antlst','antsystime',
                    'shortdata_meas','shortlst','shortsystime','shortinterp',
                    'res50data_meas','res50lst','res50systime','res50interp',
                    'res100data_meas','res100lst','res100systime','res100interp']
    
    data_dict = {} # setting up a dictionary to hold the data
    
    for i in range(len(names)):
        with open(filepath + names[i] + file_end, 'rb') as f:
            data_dict[names[i]] = np.load(f)
    
    return data_dict


# ------------------------------------------------------------------------------------- #
def LST_days_split(lst,data):
    '''
    This function splits data as a function of LST into separate arrays for each day (0 to 24h cycles).
    
    Parameters
    -----------
    lst: array of lst values for each point in data
    data: 1d or 2d data that is a function of LST in the 1st dimension (and possibly frequency in the 2nd dimension)
    
    Returns
    -----------
    lst_split: array of lst values split into each day
    data_split: data array split into each day's data
    '''
    cutoffs, _ = scipy.signal.find_peaks(lst)
    lst_split = np.split(lst,cutoffs+1,axis=0) # Note: +1 added after discovering a 1 LST index shift error
    data_split = np.split(data,cutoffs+1,axis=0) # "  "
    return lst_split, data_split


# ------------------------------------------------------------------------------------ #
def waterfall_alldays(data,lst,freqarr=freqarr_default,minfreq=30,maxfreq=200,minperbin=60,year='?',instrument='?',channel='?',source='?'):
    '''
    Plots calibration source data in waterfall plots.
    
    1. Splits data into days,
    2. Bins each day into 1h bins,
    3. For each day, makes a plot showing the data as a function of frequency for each 1h LST bin.
    
    Parameters
    ------------
    data: All unbinned power data. Is a function of LST (dimension 1) and frequency (dimension 2).
    lst: Sequential array of LST corresponding to data.
    minperbin: Width of LST bins when binning each day individually. Default: 60 mins/bin.
    source: String labelling which source is being used for the plotting, for labelling purposes. # Maybe in the future we can add automatic retrieval in the metadatabase.
    freqarr: raw frequency array: in general 0-250 MHz, 4096 channels.
    minfreq: minimum frequency at which to truncate for plotting.
    maxfreq: maximum frequency at which to truncate for plotting.
    
    Returns
    ---------
    N/A
    '''
    # Dealing with frequencies
    minfreqarg = int(minfreq/freqstep)
    maxfreqarg = int(maxfreq/freqstep)
    
    # 1. Split data into days.
    lst_split, data_split = LST_days_split(lst,data) # Looks for end of cycles in the LST array (i.e. when it goes from ~24h -> ~0h)
    
    n_days = len(lst_split) # number of days the data was split into
    
    ncols = np.min([5,n_days])
    if n_days//5 == 0 or n_days%5 == 0:
        # i.e. for 0-5 days of data
        nrows = int(n_days/ncols)
    else: 
        nrows = int(n_days/ncols+1)
    
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(4*ncols,15*nrows))
    axs = axs.flatten() # makes iteration easier
    
    # ------------ Code that ensure the colorbar is shared between all subplots and represents the full range of values ---------- #
    normalizer = Normalize(vmin = 0, vmax=np.percentile(data,95,axis=None))
    im = cm.ScalarMappable(norm=normalizer)
    ticks_cb = np.linspace(0,np.percentile(data,95,axis=None),10)
    #print(np.percentile(data,95,axis=None))
    # ---------------------------------------------- #
    
    # In this loop, i (first loop) indexes the day
    for i in range(n_days):
        
        # 2. Bin each day into 1h bins.
        data_split_binned, lst_split_bins, _  = lst_binning(data_split[i],lst_split[i],binsize=minperbin)
        
        # 3. For each 1h for this day, plot data as a function of frequency.
        FFplot, LSTplot = np.meshgrid(freqarr[minfreqarg:maxfreqarg],lst_split_bins)
        
        axs[i].pcolormesh(FFplot,LSTplot,data_split_binned[:,minfreqarg:maxfreqarg],shading='auto',norm=normalizer) # norm = normalizer
        
        axs[i].set_xlabel('Frequency [MHz]')
        axs[i].invert_yaxis()
        if (i%ncols == 0):
            axs[i].set_ylabel('LST [h]')
        ticks = np.linspace(0,24,25)
        axs[i].set_yticks(ticks=ticks)
        axs[i].set_title('Day '+str(i)+'\n'+source+' Calibration Source\n'+instrument+','+channel+','+year+'\n'+str(minperbin)+' mins/bin')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, ax=axs.ravel().tolist())
    fig.colorbar(im,cax=cbar_ax,ticks=ticks_cb,label='Absolute Fractional Difference from Time-Median')
    plt.subplots_adjust(hspace=0.15)
    plt.show()
    
    return


# ------------------------------------------------------------------------------------------------------------------------ #
# Function that plots wrt median
def time_variation(data,lst,freqarr=freqarr_default,minfreq=30,maxfreq=200,minperbin=60,year='?',instrument='?',channel='?',source='?',plot_type='waterfall'):
    
    # 1. Find median of all data along the time axis
    time_median = np.median(data,axis=0)
    #plt.plot(freqarr[minfreqarg:maxfreqarg],time_median[minfreqarg:maxfreqarg])
    #plt.show()
    
    # 2. Subtract the median from all data (-> residuals), and divide by median to get fractional difference
    res_data = np.abs((data - time_median)/time_median)
    
    # 3. Plot data either as waterfall or as regular plot
    if plot_type == 'waterfall':
        waterfall_alldays(res_data,lst,freqarr,minfreq=minfreq,maxfreq=maxfreq,minperbin=minperbin,year=year,instrument=instrument,channel=channel,source=source)
    elif plot_type == 'regular':
        plot_alldays(res_data,lst,freqarr,minperbin=minperbin,source=source)
    else:
        print('Error: plot_type must be either \'waterfall\' or \'regular\' ')
    
    return