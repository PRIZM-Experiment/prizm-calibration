import numpy as np
import prizmatoid as pzt
import data as da
import read_vna_csv as cs
from scipy import interpolate
from scipy import signal
from scipy import ndimage
import healpy as hp
from pygsm2016 import GlobalSkyModel2016


#ANALYSIS FUNCTIONS

def sky_model(GSM, A, minutes, poly, low, high):
    bin1 = round(1440/minutes)
    bins = np.where(A.any(axis=1))[0].shape[0]
    Gavg = np.nansum(GSM[:,:], axis=0)/bins
    Tdif = GSM - Gavg
    Pavg = np.nansum(A[:,:], axis=0)/bins
    Pdiff = A - Pavg
    pdifsq = Pdiff*Pdiff
    B = np.nansum(pdifsq[:,:], axis=0)
    ptdif = Pdiff*Tdif
    C = np.nansum(ptdif[:,:], axis=0)
    k = C/B
    obs = k*Pavg
    v = []
    for j in range(low, high+2, 2):
        v.append(j)
    v=np.array(v)
    fit = np.polyfit(np.log10(v/100), np.log10(np.abs(obs)), poly)    
    power = np.zeros(int((high-low)/2)+1)
    i = 0
    while i <= poly:
        power = power + fit[poly-i]*(np.log10(v/100))**i
        i += 1
    y = 10**(power)
    return y, obs, k, fit




def GSM_calibration_data(minperbin, flow, fhigh, antenna, pol, year):

    """Variables: minperbin (int), 
                  flow (even int), 
                  fhigh (even int), 
                  antenna (100 or 70), 
                  pol ('EW' or 'NS')"""
    
    freq_bins = int((fhigh-flow)/2)+1
    
    if antenna == 100 and pol == 'EW':
        with open('100MHz_EW_GSM_average' +'.npy', 'rb') as f:
            Tgsm = np.load(f)
        if year ==  2018:
            alldata = np.load('/Volumes/SSData/GSM_Cal_Power_and_Times/2018/100MHz_EW_2018.npz')
        if year ==  2021:
            alldata = np.load('/Volumes/SSData/GSM_Cal_Power_and_Times/2021/100MHz_EW_2021.npz')
        power = alldata['power']
        time = alldata['times']
        
        A = freq_binning(flow, fhigh, power)
        day_filtered = rfi_remove(A, 200, 20)
        day = antenna_binning_new(minperbin, day_filtered, time, freq_bins)
        
        return Tgsm, day
            
    

#FUNCTIONS FOR DATA RFI FLAGGING, BINNING AND SORTING

def rfi_remove(data, threshold, median_filter_window):
    """ Removes RFI data from the antenna data """   

    medians = np.median(data, axis=0)
    flattened = data - medians
    filtered = ndimage.median_filter(flattened, [1, median_filter_window])
    corrected = flattened - filtered
    MAD = np.median(np.abs(corrected - np.median(corrected)))
    x = (corrected - np.median(corrected) > threshold*MAD)
    rfi_masked_data = np.ma.masked_array(data, x)
   
    return rfi_masked_data


def freq_binning(low, high, dayshift):
    """ Sorts the antenna data into frequency bins matching the GSM (usually 2MHz) and cuts out frequencies that are higher or lower than GSM """
    
    freqlow = []
    freqhigh = []
    for j in range(low-1, high+1, 2):
        freqlow.append(int(4096*j/250))
    for j in range(low+1, high+3, 2):
        freqhigh.append(int(4096*j/250))
    A = []
    for k in range(0, int((high-low)/2)+1):
        pdat = np.sum(dayshift[:, freqlow[k]:freqhigh[k]], axis = 1)/(freqhigh[k] - freqlow[k])
        A.append(pdat)
    A1 = np.array(A)
    A2 = A1.T
    return A2


def align_GSMdata(minutes, GSMdata):
    bin1 = round(1440/minutes)
    I1 = int((360/720)*bin1)
    I2 = int((530/720)*bin1)
    TGSM = []
    zerobin = (bin1 - I2) + I1
    for i in range(zerobin, bin1):
        T2 = GSMdata[i,:].tolist()
        TGSM.append(T2)
    for i in range(0, zerobin):
        T2 = GSMdata[i,:].tolist()
        TGSM.append(T2)
    TGSM = np.array(TGSM)
    return TGSM



def GSM_add_zeros(TGSM, data, minperbin):
    """ Adds zeros to GSM when each antenna data set has no data """

    bin1=int(1440/minperbin)
    GSM = np.zeros((bin1, TGSM.shape[1]))
    on = np.where(data.any(axis=1))[0]
    for i in on:
        GSM[i,:] = TGSM[i,:]
    xx = np.isnan(data)
    for i in range(0, xx.shape[0]):
        for j in range(0, xx.shape[1]):
            if xx[i,j]:
                GSM[i,j] = np.nan 
    return GSM


def antenna_binning_new(minperbin, data, totaltime, freq_bins):

    #convert LST to minutes in day
    minutes = totaltime*1440/23.94
    bins = int(1440/minperbin)
    bin_num = np.zeros(len(minutes))
    for i, minute in enumerate(minutes):
        bin_num[i] = np.floor(minute/minperbin)
    freq_list = [i for i in range(0, freq_bins)]
        
    binned = np.zeros((bins, freq_bins))
    for j in range(0, bins):
        cond = bin_num == j
        mesh = np.ix_(cond, freq_list)
        correct_data = data[mesh] 
        binned[j,:] = np.sum(correct_data, axis=0)/len(correct_data)
    binned[np.isnan(binned)] = 0
    
    return binned


def antenna_binning(minperbin, data, totaltime, freq_bins):
    
    bin_starts = [i for i in range(0, 1440, minperbin)]
    bin_starts.append(1440)

    minutes = totaltime*1440/23.94
    start_time = minutes.min()
    end_time = minutes.max()
    first_bin = int((np.floor(start_time/minperbin))*minperbin)
    last_bin = int((np.ceil(end_time/minperbin))*minperbin)

    bin_starts_data = []
    for i in range(first_bin, last_bin, minperbin):
        bin_starts_data.append(i)
    bin_starts_data.append(last_bin)

    binnum = np.zeros(len(minutes))
    for i in range(0, len(minutes)):
        for j in range(0, len(bin_starts)-1):
            if minutes[i] >= bin_starts[j] and minutes[i] < bin_starts[j+1]:
                binnum[i] = j

    binned = np.zeros((len(bin_starts)-1, freq_bins))
    for i in range(0, len(bin_starts)-1):
        tmp = []
        for j in range(0, len(minutes)):
            if binnum[j] == i:
                tmp.append(data[j,:])
            binned[i,:] = np.sum(tmp, axis=0)/len(tmp)

    binned[np.isnan(binned)] = 0

    return binned




#FUNCTIONS TO CREATE GSM TEMP ARRAYS

def healpy_beam(beam_dict, pol, healpy_nside=256, site_latitude=-46.88694):
    """ Converts a beam simulation dictionary into HealPix format.

    Given an input dictionary `beam_dict` containing the raw beam simulation and
    associated spherical coordinates, generates a new dictionary is generated in
    which the beam amplitudes and spherical coordinates are adapted to the
    HealPix format with pixelization set by `healpy_nside`.

    Args:
        beam_dict: a dictionary containing a raw beam simulation.
        healpy_nside: an integer specipying the HealPix pixelization.
        site_latitude: the latitute of the instrument associated with the beam.

    Returns:
        A dictionary containing the sampled HealPix beam amplitudes and
        associated spherical coordinates for each frequency channel (in MHz). A
        typical output returned by this function would have the following
        structure.

        {
        'theta': numpy.array,
        'phi': numpy.array,
        20: numpy.array,
        22: numpy.array,
        ...,
        198: numpy.array,
        200: numpy.array,
        'normalization': numpy.array,
        }
    """

    # Initializes the dictionary which will hold the HealPy version of the beam.
    healpy_beam_dict = {}

    # Extracts the frequencies for which beams are available in `beam_dict`.
    frequencies = [key for key in beam_dict.keys() if isinstance(key, float)]
    n_freq = len(frequencies)
    
    # Initializes a HealPy pixelization and associated spherical coordinates.
    healpy_npix = hp.nside2npix(healpy_nside)
    healpy_theta, healpy_phi = hp.pix2ang(healpy_nside,
                                              np.arange(healpy_npix))

    # Stores spherical coordinates in `healpy_beam_dict`.
    healpy_beam_dict['theta'] = healpy_theta
    healpy_beam_dict['phi'] = healpy_phi

    # SciPy 2D interpolation forces us to do proceed in chunks of constant
    # coordinate `healpy_theta`. Below we find the indices at which
    # `healpy_theta` changes.
    indices = np.where(np.diff(healpy_theta) != 0)[0]
    indices = np.append(0, indices + 1)

    # Initializes the NumPy array which will contain the normalization factor
    # for each beam.
    beam_norms = np.zeros(n_freq)

    # Loops over the different frequencies for which the beam has been
    # simulated.
    for i, frequency in enumerate(frequencies):

        # Computes the actual beam from the information contained in
        # `beam_dict`.
        beam = 10**(beam_dict[frequency]/10)

        # Interpolates beam.
        beam_interp = interpolate.interp2d(beam_dict['theta'],
                                           beam_dict['phi'],
                                           beam,
                                           kind='cubic',
                                           fill_value=0)

        # Initializes `healpy_beam`, the HealPy version of the beam.
        healpy_beam = np.zeros(len(healpy_theta))

        # Constructs the HealPy beam.
        for j in range(np.int(len(indices)/2) + 2):
            start = indices[j]
            end = indices[j+1]
            healpy_beam[start:end] = beam_interp(healpy_theta[start],
                                             healpy_phi[start:end])[:,0]

        # Fills `beam_norms` with the appropriate normalization factors for
        # each HealPy beam.
        beam_norms[i] = np.sqrt(np.sum(healpy_beam**2))
        
        # Rotates and stores the the HealPy beam in the `healpy_beam_dict` under
        # the appropriate frequency entry.
        if pol == 'NS':
            beam_rotation = hp.rotator.Rotator([0, 0, 90 - site_latitude])
        if pol == 'EW':
            beam_rotation = hp.rotator.Rotator([90, 0, 90 - site_latitude])
        healpy_beam = beam_rotation.rotate_map_pixel(healpy_beam/beam_norms[i])
        healpy_beam_dict[frequency] = healpy_beam

    # Adds the beam normalizations as a separate entry in `heapy_beam_dict`.
    healpy_beam_dict['normalization'] = beam_norms

    # Returns the HealPy version of the beam in a dictionary format.
    return healpy_beam_dict



def save_GSM_maps(flow, fhigh):
    gsm_2016 = GlobalSkyModel2016(freq_unit='MHz')

    for i in range(flow,fhigh+2,2):
        gsm_map = gsm_2016.generate(i)
        gsm_map_eq = change_coord(gsm_map, ['G', 'C'])
        gsm_map_lowres = hp.ud_grade(gsm_map_eq, 8, order_in = 'RING', order_out = 'RING')
        np.save('gsm' + 'i', gsm_map_lowres)
        
        

def get_GSM_temps(healpy_beam, flow, fhigh, minperbin):
    
    temperatures1 =[]
    gsm_2016 = GlobalSkyModel2016(freq_unit='MHz')

    for i in range(flow,fhigh+2,2):
        gsm_map = gsm_2016.generate(i)
        gsm_map_eq = change_coord(gsm_map, ['G', 'C'])
        gsm_map_lowres = hp.ud_grade(gsm_map_eq, 256, order_in = 'RING', order_out = 'RING')
        nside=256
        alm_map_eq = hp.map2alm(gsm_map_lowres)
        alm_BEAM = hp.map2alm(healpy_beam[i])
        temp_map = np.full(gsm_map_lowres.size, 1)
        alm_temp_map = hp.map2alm(temp_map)
        integral_beam0 = np.real(np.sum(alm_temp_map*alm_BEAM))
        lmax = np.int(np.round(np.sqrt(2*len(alm_BEAM) - 0.5)))  ;
        m = np.zeros(len(alm_BEAM));


        icur = 0;
        for i in range(0, lmax):
            nn = lmax - i;
            m[icur:icur+nn] = i;
            icur = icur+nn
  

        #phi_rot1 = np.linspace(0,2*np.pi,1440/minperbin)
        phi_rot1 = np.linspace(0,2*np.pi,(1440/minperbin)+1)
        phitmp = phi_rot1.tolist()
        phitmp.pop()
        phi_rot1 = np.array(phitmp)

        temperatures0 = []
        for phi in phi_rot1:
    
            new_alm_beam = alm_BEAM*np.exp(-1j*phi*m)
    

 
            y0 = new_alm_beam*np.conj(alm_map_eq)

            integral_beam_map0 = np.real((np.sum(y0[:lmax])+2*np.sum(y0[lmax:])))



            amp_alm_space0 = integral_beam_map0/integral_beam0
    
            temperatures0.append(amp_alm_space0)

        temperatures1.append(temperatures0)
    
    temp = np.array(temperatures1)
    T = temp.T

    return T


def change_coord(m, coord):
    """ Change coordinates of a HEALPIX map

    Parameters
    ----------
    m : map or array of maps
      map(s) to be rotated
    coord : sequence of two character
      First character is the coordinate system of m, second character
      is the coordinate system of the output map. As in HEALPIX, allowed
      coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

    Example
    -------
    The following rotate m from galactic to equatorial coordinates.
    Notice that m can contain both temperature and polarization.
    >>>> change_coord(m, ['G', 'C'])
    """
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))

    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]







#FUNCTIONS TO CREATE 'GOOD' DATA PRODUCTS TO BE SAVED -- SHOULD BE MOVED SOMEWHERE ELSE!
        
def run_data(prizm_data, antenna, polarization, eff = None):
    """ A bunch of functions put into one function to save space on Notebook """
        
    #makes the switch flags 
    da.add_switch_flags(prizm_data, instruments =[antenna], channels = [polarization])
 

    #finds the indices of each mode
    select_antenna = pzt.shrink_flag(prizm_data[antenna][polarization]['Flags']['antenna'], (1,1))
    
    select_short = pzt.shrink_flag(prizm_data[antenna][polarization]['Flags']['short'], (1,1))
    
    #Gets data to simplify functions below
    data = prizm_data[antenna][polarization]['pol']


    #finding the times
    data_time = prizm_data[antenna][polarization]['time_sys_start'] - prizm_data[antenna][polarization]['time_sys_start'][0]
    
    
    short_ind=[i for i, x in enumerate(select_short) if x]
    antenna_ind=[i for i, x in enumerate(select_antenna) if x]
    
    #Don't need this anymore, I think
    shorton = [x for x in short_ind if x <= data.shape[0]-1]
    antennaon = [x for x in antenna_ind if x <= data.shape[0]-1]
    
    #Interpolate short to get value at each antenna time. If this works, we no longer need "interpolate_short" or "find_short" or "find_antenna_times" function
    short_vals = np.zeros((data[antennaon].shape[0], data[antennaon].shape[1]))
    for j in range(0, data.shape[1]):
        short_vals[:,j] = interpolate.interp1d(data_time[shorton], data[shorton, j], fill_value="extrapolate")(data_time[antennaon])
    
    if eff:
        p = (data[antennaon] - short_vals)/(eff)
    else:
        p = data[antennaon] - short_vals
    
    return p, antennaon



def find_efficiency1(ant_s11, xsmooth):
    """ Finds the antenna efficiency from the antenna s11 only. """
    
    #read the sll files
    s11=cs.read_vna_file(ant_s11)
    s11freqs = s11[:,0]
    
    #convert the s11 from dB to linear 
    s11_eff =[0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_eff[i]= 10**(s11[i,1]/20)
   
    #interpolate and smooth efficiency (to match frequencies used in data)
    sos = signal.butter(1, xsmooth, btype='lp', output='sos')
    lin=interpolate.interp1d(s11freqs, signal.sosfilt(sos, s11_eff), kind='slinear',fill_value="extrapolate")
    xnew= np.linspace(0,250000000, num = 4096, endpoint= True)
    eff=1-((lin(xnew))**2)
    
    return eff
  


def find_efficiency2(ant_s11, frontend_s11, xsmooth):
    """ Finds the antenna efficiency from the antenna and front end s11 data. """
    
    #read the sll files
    s11=cs.read_vna_file(ant_s11)
    j6=cs.read_vna_file(frontend_s11)
    s11freqs = s11[:,0]
    
    #convert the angles to radians and amplitudes from dB to linear
    s11_ang = [0 for x in range((len(s11[:,0])))] 
    j6_ang = [0 for x in range((len(s11[:,0])))] 
    s11_eff =[0 for x in range((len(s11[:,0])))] 
    j6_eff =[0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_ang[i] = s11[i,2]*(np.pi/180)
        j6_ang[i] = j6[i,2]*(np.pi/180)
        s11_eff[i]= 10**(s11[i,1]/20)
        j6_eff[i]= 10**(j6[i,1]/20)
    
    #find the real and imaginary parts of the s11 data
    s11_real = [0 for x in range((len(s11[:,0])))] 
    s11_imag = [0 for x in range((len(s11[:,0])))] 
    j6_real = [0 for x in range((len(s11[:,0])))] 
    j6_imag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_real[i]= s11_eff[i]*np.cos(s11_ang[i])
        s11_imag[i]= s11_eff[i]*np.sin(s11_ang[i])
        j6_real[i]= j6_eff[i]*np.cos(j6_ang[i])
        j6_imag[i]= j6_eff[i]*np.sin(j6_ang[i])
        
    #convert the reflection coefficients (s11) to impedences (Z)    
    Z_s11_real = [0 for x in range((len(s11[:,0])))] 
    Z_s11_imag = [0 for x in range((len(s11[:,0])))] 
    Z_j6_real = [0 for x in range((len(s11[:,0])))] 
    Z_j6_imag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        Z_s11_imag[i] =(((50*s11_imag[i])*(1-s11_real[i]))-((50+(50*s11_real[i]))*(0-s11_imag[i])))/(((1-s11_real[i])**2)+((0-s11_imag[i])**2))
        Z_s11_real[i] =(((50+(50*s11_real[i]))*(1-s11_real[i])) + ((50*s11_imag[i])*(0-s11_imag[i])))/(((1-s11_real[i])**2)+((0-s11_imag[i])**2))
        Z_j6_real[i] = (((50+(50*j6_real[i]))*(1-j6_real[i])) + ((50*j6_imag[i])*(0-j6_imag[i])))/(((1-j6_real[i])**2)+((0-j6_imag[i])**2))
        Z_j6_imag[i] = (((50*j6_imag[i])*(1-j6_real[i]))-((50+(50*j6_real[i]))*(0-j6_imag[i])))/(((1-j6_real[i])**2)+((0-j6_imag[i])**2))
        
   #find the top and bottom of the total reflection co-efficent (gamma) after combining the antenna and front end     
    topreal = [0 for x in range((len(s11[:,0])))]
    botreal = [0 for x in range((len(s11[:,0])))]
    topimag = [0 for x in range((len(s11[:,0])))]
    botimag = [0 for x in range((len(s11[:,0])))]
    for i in range(len(s11[:,0])):
        topreal[i] = Z_s11_real[i] - Z_j6_real[i]
        botreal[i] = Z_s11_real[i] + Z_j6_real[i]
        topimag[i] = Z_s11_imag[i] + Z_j6_imag[i]
        botimag[i] = Z_s11_imag[i] + Z_j6_imag[i]
        
    #find the real and imaginary parts of gamma   
    gammareal = [0 for x in range((len(s11[:,0])))]
    gammaimag = [0 for x in range((len(s11[:,0])))]
    for i in range(len(s11[:,0])):
        gammareal[i] = ((topreal[i]*botreal[i])+(topimag[i]*botimag[i]))/((botreal[i]**2)+(botimag[i]**2))
        gammaimag[i] = ((topimag[i]*botreal[i])-(topreal[i]*botimag[i]))/((botreal[i]**2)+(botimag[i]**2))
    
    mag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        mag[i] = np.sqrt((gammareal[i])**2 + (gammaimag[i])**2)
   
    #interpolate and smooth gamma to match frequencies used in data
    sos = signal.butter(1, xsmooth, btype='lp', output='sos')
    lingamma=interpolate.interp1d(s11freqs, signal.sosfilt(sos, mag), kind='slinear',fill_value="extrapolate")
    xnew= np.linspace(0,250000000, num = 4096, endpoint= True)
    eff=1-((lingamma(xnew))**2)
    
    return eff


def find_short(shorton, prizm_data, newlist_antend, antenna, polarization):

    somelist1 = shorton
    newlist_shortend = []

    for i in range(len(somelist1)-1):
        if somelist1[i+1] != somelist1[i] + 1 :
            newlist_shortend.append(somelist1[i])
    
    x=len(somelist1)-1        
    newlist_shortend.append(somelist1[x])        
    
    newlist_shortstart = []

    for i in range(1, len(somelist1)):
        if somelist1[i-1] != somelist1[i] - 1:
            newlist_shortstart.append(somelist1[i])
         
    newlist_shortstart.insert(0,somelist1[0])
    
    print(len(newlist_shortstart))

    short_lengths = list(np.array(newlist_shortend) - np.array(newlist_shortstart))
    short = []
    for i in range(0, len(newlist_shortend)):
        sum = np.zeros(4096)
        for j in range(0, short_lengths[i]+1):
            sum = sum + prizm_data[antenna][polarization][newlist_shortstart[i]+j]
        short.append(sum/(short_lengths[i]+1))
     
       
    dif=[]
    for i in range(0, len(newlist_shortend)-1):
        dif.append(newlist_shortend[i+1] - newlist_shortend[i])
    base = np.average(dif)
  
    for i in range(0, len(dif)):
        if dif[i] > base + base/2:
            short.insert(i+1, pzt.interpolate_short(prizm_data, antenna, polarization)[i])
 
   
    if newlist_shortend[-1] < newlist_antend[-1]:
        short.append(pzt.interpolate_short(prizm_data, antenna, polarization)[-1])
    
    
    return short
    



def find_antenna_times(antennaon): 
    somelist = antennaon
    newlist_antend = []

    for i in range(0, len(somelist)-1):
        if somelist[i+1] != somelist[i] + 1:
            newlist_antend.append(somelist[i])
    
    x=len(somelist)-1        
    newlist_antend.append(somelist[x])
    print('There are ' +str(len(newlist_antend))+ ' end times')
    print(newlist_antend)

    somelist = antennaon
    newlist_antstart = []
    newlist_antstart.append(somelist[0])


    for i in range(1, len(somelist)):
        if somelist[i-1] != somelist[i] - 1:
            newlist_antstart.append(somelist[i])
   
    
    print('There are ' +str(len(newlist_antstart))+ ' start times')
    print(newlist_antstart)
    
    return newlist_antstart, newlist_antend 

