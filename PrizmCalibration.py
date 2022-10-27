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
    


