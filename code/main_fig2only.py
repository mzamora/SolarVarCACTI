#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:13:10 2022

@author: Mónica Zamora Z, DIMEC-UChile
         mzamora.github.io
"""

# Fig 2

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np
import numpy.matlib
from netCDF4 import Dataset
import pvlib as pvlib
from os import listdir
from datetime import datetime, timedelta
from scipy.interpolate import interp1d


file_dir= '../data/'
prfx=['corqcrad1longM1.c2','cormwrret1liljclouM1.c2',
      'cormfrsrcldod1minM1.c1','corceilM1.b1',
      'corarsclkazrbnd1kolliasM1.c1','cor30smplcmask1zwangM1.c1']
lss=listdir(file_dir)

# set date
d0=datetime(2018,10,1) #official start date https://www.arm.gov/research/campaigns/amf2018cacti
d1=datetime(2019,4,30) #official end date
#thisday=datetime(2019,3,1)
#dates=np.arange(d0,d0+timedelta(days=16),timedelta(days=1)).astype(datetime)
dates=np.arange(d0,d1,timedelta(days=1)).astype(datetime)

#data outputs
tout=np.array([],dtype=datetime)
ghiout=np.array([]); difout=np.array([]); dniout=np.array([])
ghicsout=np.array([]); difcsout=np.array([]); dnicsout=np.array([])
ktout=np.array([]); ktmeanout=np.empty([0,3]); ktstdout=np.empty([0,3])
eleout=np.array([]); szaout=np.array([])
zblyrout=np.empty([0,10]); ztlyrout=np.empty([0,10]); nlyrout=np.array([])
precipout=np.array([])
df1min=pd.DataFrame(); df5min=pd.DataFrame(); df15min=pd.DataFrame(); 
df30min=pd.DataFrame(); df60min=pd.DataFrame()

thisday=dates[112] # to just create plots for one day (Fig 2)

# beginning of the filename to search for
nm=[prf+thisday.strftime('.%Y%m%d') for prf in prfx]
    
# kepp going only if both radiation and kzar data are available
#, and there's no ghi missing data
matchkzr = [match for match in lss if nm[4] in match]
matchrad = [match for match in lss if nm[0] in match]
ds=Dataset(file_dir+matchrad[0])
ghiok=np.sum(ds['BestEstimate_down_short_hemisp'][:].data==-9999)==0
if len(matchkzr)==0 or len(matchrad)==0 or (not ghiok):
    print('Data missing or missing ghi for '+thisday.strftime('%Y%m%d'))

# 0 is RAD data
matches = [match for match in lss if nm[0] in match]
ds=Dataset(file_dir+matches[0])
tghi=ds['time'][:].data
ghi=ds['BestEstimate_down_short_hemisp'][:].data
dif=ds['down_short_diffuse_hemisp'][:].data
dni=ds['short_direct_normal'][:].data

# 1 is LWP data from microwave radiometers doi:10.1109/TGRS.2007.903
matches = [match for match in lss if nm[1] in match]
ds=Dataset(file_dir+matches[0])
tlwp=ds['time'][:].data
lwp=ds['be_lwp'][:].data #in g/m2
pwv=ds['be_pwv'][:].data #precipitable water wapor (cm)

# 2 is COD data
matches = [match for match in lss if nm[2] in match]
ds=Dataset(file_dir+matches[0])
tcod=ds['time'][:].data
cod=ds['optical_depth_instantaneous'][:].data
reff=ds['effective_radius_instantaneous'][:].data

# 3 is ceilometer data
matches = [match for match in lss if nm[3] in match]
ds=Dataset(file_dir+matches[0])
tceil=ds['time'][:].data
zb1ceil=ds['first_cbh'][:].data
zb2ceil=ds['second_cbh'][:].data
zb3ceil=ds['third_cbh'][:].data

# 4 is cloud in time and space Kollias' algorithm
matches = [match for match in lss if nm[4] in match]
ds=Dataset(file_dir+matches[0])
tkazr=ds['time'][:].data
zbkazr=ds['cloud_base_best_estimate'][:].data
zblyr=ds['cloud_layer_base_height'][:][:].data #time and layer
ztlyr=ds['cloud_layer_top_height'][:][:].data # time and layer
nlyrs=np.sum(zblyr>0,axis=1).astype(np.float)

# clear sky irradiance
latitude, longitude, tz, altitude, name = -32.12641,-64.72837,'UTC',1141,'Cordoba'
times = pd.date_range(start=thisday.strftime('%Y-%m-%d'), end=thisday.strftime('%Y-%m-%d 23:59'), freq='1Min', tz=tz)
solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
apparent_zenith = solpos['apparent_zenith']
airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
pressure = pvlib.atmosphere.alt2pres(altitude)
airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
LT0 = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)[0]
dni_extra = pvlib.irradiance.get_extra_radiation(times)
# an input is a pandas Series, so solis is a DataFrame
ineichen = pvlib.clearsky.ineichen(apparent_zenith, airmass, LT0, altitude, dni_extra)
ghics=ineichen['ghi'].values
kt=np.zeros(ghics.shape)
ele0=(solpos['elevation']>0).values
fel=(solpos['elevation']>20).values #where elevation is greater that 10 deg
t0=tghi[fel][0] # start time for ele 10
t1=tghi[fel][-1] # end time for ele 10
kt[ele0]=ghi[ele0]/ghics[ele0]
kt=pd.DataFrame(kt)
kts=kt.rolling(5,center=True).std().squeeze()
ktm=kt.rolling(5,center=True).mean().squeeze()
fclear=np.logical_and(np.logical_and(kts/ktm<0.05,np.abs(ktm-1)<0.15),fel)
#plt.plot(tghi/3600,ghi,tghi/3600,ghics)
# find a better linke turbidity
niter=0
eps=100
while eps>10:
    ineichen = pvlib.clearsky.ineichen(apparent_zenith, airmass, LT0, altitude, dni_extra)
    ghics=ineichen['ghi'].squeeze()
    kt=np.zeros(ghics.shape)
    kt[ele0]=ghi[ele0]/ghics[ele0]
    ktbias=kt[fclear]-1
    eps=np.sum(np.abs(ktbias))
    print(eps)
    LT0=LT0-np.sign(ktbias.mean())*0.05 # lower/higher LT needed
    print(LT0)
    niter=niter+1
    if niter>1000: break
#plt.plot(tghi/3600,ghics)
        
kt=pd.DataFrame(kt) #needed to calculate the rolling things
twindows=[5,15,30,60]
ktmean=np.zeros([len(tghi),4])
ktstd=np.zeros([len(tghi),4])
    
# plot and calculate rolling mean and std for different time windows
fig1,ax1=plt.subplots(5,1,figsize=[5,8])
ax1[0].plot(tghi/3600,ghi,tghi/3600,ghics)
ax1[0].set_ylabel('GHI (W/m$^2$)')
ax1[0].legend(['Measured','Clear sky'],loc='upper left',prop={'size':8}); 
ax1[0].set_xlim([9.5,23.5]); ax1[0].set_ylim([0,1500])
ax1[1].plot(tghi[fel]/3600,kt[fel].values,[9.5,23.5],[1,1],'r--'); 
ax1[1].set_ylabel('$k_t$'); ax1[1].set_xlim([9.5,23.5])
for ip in range(len(twindows)):
    twindow=twindows[ip]
    ktmean[:,ip]=kt.rolling(twindow,center=True).mean().squeeze()
    ktstd[:,ip]=kt.rolling(twindow,center=True).std().squeeze()
    ax1[2].plot(tghi[fel]/3600,ktstd[fel,ip]/ktmean[fel,ip])
ax1[2].set_ylabel('$\sigma_{k_t} / \overline{k_t}$')
ax1[2].legend(['5 min','15 min','30 min','1 h'],prop={'size':8})

precipkazr=np.zeros(len(tkazr)) # it might be raining when zb is at the surface?
# plot the clouds!! in t and z
# only when elevation greater than 10 deg
fel2=np.logical_and(tkazr>t0,tkazr<t1)
for it in np.where(fel2)[0]:
    zbs=zblyr[it,:]
    if np.sum(zbs>0)==0: continue
    zts=ztlyr[it,:]
    for ib in range(0,len(zbs)):
        if zbs[ib]<=0: continue
        ax1[3].plot([tkazr[it]/3600,tkazr[it]/3600],[zbs[ib]/1000,zts[ib]/1000],'k')
        if zbs[ib]==160:
            precipkazr[it]=True
ax1[3].set_ylabel('Cloud location (km)');
ax1[3].plot([9.5,23.5],[2,2],'r--');
ax1[3].plot([9.5,23.5],[6,6],'r--'); 
ax1[3].text(9.8,0.8,'Low',font={'size':8})
ax1[3].text(9.8,4,'Mid',font={'size':8})
ax1[3].text(9.8,7.5,'High',font={'size':8})
# number of layers
inodata=np.where(zblyr[:,0]==-9999)[0] # fill missing data with nans
nlyrs[inodata]=np.nan
# add in red times when data were not available
ax1[3].plot(np.matlib.repmat(tkazr[inodata]/3600,2,1),np.matlib.repmat([0,15000],len(inodata),1).T,'r')
ax1[4].plot(tkazr/3600,nlyrs); 
ax1[4].set_ylabel('Cloud layers'); ax1[4].set_xlim([9.5,23.5])
ax1[4].set_xlabel('UTC Time (h)')

for ip in range(4):
    ax1[ip].set_xticks([]);
    ax1[ip].set_xlim([9.5,23.5])
plt.tight_layout()

plt.figtext(0.17,0.98, 'a)', size=12)
plt.figtext(0.17,0.795, 'b)', size=12)
plt.figtext(0.17,0.605, 'c)', size=12)
plt.figtext(0.17,0.42, 'd)', size=12)
plt.figtext(0.17,0.235, 'e)', size=12)

fig1.savefig('../out/Fig2_20190121.png'); plt.close(fig1)
    
