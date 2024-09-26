import numpy as np
from matplotlib import pyplot as plt

datadir = '../test_data/2021/100MHz/NS/'

def make_acf(dat,t,dt,tmax):
    n=int(tmax/dt)
    tot=np.zeros(n)
    wt=np.zeros(n)
    nn=len(dat)
    for i in range(nn):
        for j in range(i+1,nn):
            delt=np.abs(t[i]-t[j])
            k=int(delt/dt) 
            if k<n:
                tot[k]=tot[k]+dat[i]*dat[j]
                wt[k]=wt[k]+1
            else:
                break
    return tot,wt

def make_acf_alt(dat,t,dt,tmax):
    n=int(tmax/dt)
    tot=np.zeros(n)
    wt=np.zeros(n)
    nn=len(dat)
    for i in range(nn):
        # used to be range(i+1,nn) below, changed for testing
        for j in range(i,nn):
            delt=np.abs(t[i]-t[j])
            k=int(delt/dt)  # this is the change
            if k<n:
                tot[k]=tot[k]+dat[i]*dat[j]
                wt[k]=wt[k]+1
            else:
                break
    return tot,wt

def make_acf_v3(dat,t,dt,tmax):
    
    
    
    
    
    return

plt.ion()


# dat=np.load(datadir+'shortdata_meas_2021_100MHz_NS.npy')
# lst=np.load(datadir+'shortlst_2021_100MHz_NS.npy')
# t=np.load(datadir+'shortsystime_2021_100MHz_NS.npy')

# tmax=1.6374e9
# tmin=1.6365e9

# mask=(t>tmin)&(t<tmax)
# tt=t[mask]
# dd=dat[mask,:]
# ll=lst[mask]

# dt=600
# tot,wt=make_acf(dd[:,1000]-dd[:,1000].mean(),tt,dt=dt,tmax=2*86400)
# tvec=np.arange(len(tot))*dt
# mm=wt>30
# plt.clf()
# plt.plot(tvec[mm]/3600,(tot[mm]/wt[mm]),'.')
# plt.show()