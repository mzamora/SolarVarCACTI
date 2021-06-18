#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:41:11 2021

@author: monica
"""

import pandas as pd
import matplotlib.pyplot as plt
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

for thisday in dates:
    # beginning of the filename to search for
    nm=[prf+thisday.strftime('.%Y%m%d') for prf in prfx]
        
    # kepp going only if both radiation and kzar data are available
    matchkzr = [match for match in lss if nm[4] in match]
    matchrad = [match for match in lss if nm[0] in match]
    if len(matchkzr)==0 or len(matchrad)==0:
        print('Data missing for '+thisday.strftime('%Y%m%d'))
        continue
    
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
    twindows=[5,15,30]
    ktmean=np.zeros([len(tghi),3])
    ktstd=np.zeros([len(tghi),3])
    # plot and calculate rolling mean and std for different time windows
    fig1,ax1=plt.subplots(2,3,figsize=[12,6])
    ax1[0][0].plot(tghi/3600,ghi,tghi/3600,ghics)
    ax1[0][0].set_xlabel('UTC Time (h)'); ax1[0][0].set_ylabel('GHI (W/m$^2$)')
    ax1[0][0].legend(['Obs.','Clear sky'],loc='upper left'); 
    ax1[0][0].set_xlim([8,24]); ax1[0][0].set_ylim([0,1400])
    ax1[0][1].plot(tghi[fel]/3600,kt[fel].values,[8,24],[1,1],'r--'); 
    ax1[0][1].set_xlabel('UTC Time (h)'); ax1[0][1].set_ylabel('$k_t$'); ax1[0][1].set_xlim([8,24])
    for ip in range(len(twindows)):
        twindow=twindows[ip]
        ktmean[:,ip]=kt.rolling(twindow,center=True).mean().squeeze()
        ktstd[:,ip]=kt.rolling(twindow,center=True).std().squeeze()
        ax1[0][2].plot(tghi[fel]/3600,ktstd[fel,ip]/ktmean[fel,ip])
        ax1[1][ip].plot(ktmean[fel,ip],ktstd[fel,ip],'.')
    ax1[0][2].set_xlabel('UTC Time (h)'); ax1[0][2].set_ylabel('$\sigma_{k_t} / \overline{k_t}$')
    ax1[0][2].legend(['5 min','15 min','30 min'])
    for ip in range(len(twindows)):
        ax1[1][ip].set_ylabel('$\sigma_{k_t}$'); ax1[1][ip].set_xlabel('$\overline{k_t}$')
        ax1[1][ip].set_title('$\Delta t=$ '+str(twindows[ip])+' min'); 
        ax1[1][ip].set_xlim([0,1.3]); ax1[1][ip].set_ylim([0,0.5])
    plt.suptitle(thisday.strftime('%Y/%m/%d')); plt.tight_layout()
    fig1.savefig('../out/ghikt_'+thisday.strftime('%Y%m%d')+'.png'); plt.close(fig1)
    
    # number of layers
    inodata=np.where(zblyr[:,0]==-9999)[0] # fill missing data with nans
    nlyrs[inodata]=np.nan
    fig1,ax1=plt.subplots(3,1,figsize=[9,5])
    ax1[0].plot(tkazr/3600,nlyrs); #ax1[0].set_xlabel('UTC Time (h)')
    ax1[0].set_ylabel('#Cloud layers'); ax1[0].set_xlim([8,24])
    ax1[0].set_xticks([])
    
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
            ax1[1].plot([tkazr[it]/3600,tkazr[it]/3600],[zbs[ib],zts[ib]],'k')
            if zbs[ib]==160:
                precipkazr[it]=True
    ax1[1].set_ylabel('Cloud location (m)'); ax1[1].set_xlim([8,24]); #ax1[1].set_xlabel('UTC Time (h)')
    # add in red times when data were not available
    ax1[1].set_xticks([])
    ax1[1].plot(np.matlib.repmat(tkazr[inodata]/3600,2,1),np.matlib.repmat([0,15000],len(inodata),1).T,'r')
    
    ax1[2].plot(tkazr/3600,precipkazr)
    ax1[2].set_xlabel('UTC Time (h)'); ax1[2].set_ylabel('Precipitation?')
    ax1[2].set_xlim([8,24])
    ax1[2].set_yticks([0,1]); ax1[2].set_yticklabels(['No','Yes']); 
    plt.suptitle(thisday.strftime('%Y/%m/%d')); plt.tight_layout()
    fig1.savefig('../out/clouds_'+thisday.strftime('%Y%m%d')+'.png'); plt.close(fig1)

    # convert cloud variables (4 s) to tghi (1 min) resolution
    fnlyrs=interp1d(tkazr,nlyrs,kind='nearest') #to lookup from tkzar to tghi at closest time
    nlyrs1min=fnlyrs(tghi)
    zblyr1min=np.zeros([len(tghi),10])
    ztlyr1min=np.zeros([len(tghi),10])
    for ilyr in range(0,10):
        fzb=interp1d(tkazr,zblyr[:,ilyr],kind='nearest') #goes with tkzar
        zblyr1min[:,ilyr]=fzb(tghi)
        fzt=interp1d(tkazr,ztlyr[:,ilyr],kind='nearest') #goes with tkzar
        ztlyr1min[:,ilyr]=fzt(tghi)
    precip1min=np.zeros(len(tghi))
    precip1min[zblyr1min[:,0]==160]=1 #
    
    # append variables to continue loop. only export when elevation>10
    tout=np.concatenate((tout, np.array([thisday+timedelta(seconds=ti) for ti in tghi[fel]])))
    ghiout=np.concatenate((ghiout, ghi[fel]))
    difout=np.concatenate((difout, dif[fel]))
    dniout=np.concatenate((dniout, dni[fel]))
    ghicsout=np.concatenate((ghicsout, ineichen['ghi'].values[fel]))
    difcsout=np.concatenate((difcsout, ineichen['dhi'].values[fel]))
    dnicsout=np.concatenate((dnicsout, ineichen['dni'].values[fel]))
    ktout=np.concatenate((ktout, kt.values[fel].squeeze()))
    ktmeanout=np.concatenate((ktmeanout, ktmean[fel,:]))
    ktstdout=np.concatenate((ktstdout, ktstd[fel,:]))
    eleout=np.concatenate((eleout, solpos['elevation'].values[fel]))
    szaout=np.concatenate((szaout, solpos['zenith'].values[fel]))
    zblyrout=np.concatenate((zblyrout,zblyr1min[fel,:]))
    ztlyrout=np.concatenate((ztlyrout,ztlyr1min[fel,:]))
    nlyrout=np.concatenate((nlyrout,nlyrs1min[fel]))
    precipout=np.concatenate((precipout,precip1min[fel]))
    
    
'''
Now data is ready to be analyzed 
'''  

fig,ax=plt.subplots(1,1)
binss=np.linspace(0,2,21)
ax.hist(ktout,bins=binss,alpha=0.5)  
ax.hist(ktout[precipout==0],bins=binss,alpha=0.5) 
ax.hist(ktout[precipout==1],bins=binss,alpha=0.5)  
ax.set_xlabel('$k_t$')
ax.set_ylabel('Frequency');
ax.legend(['All data','No Precip.','Precip.'])
plt.tight_layout(); 
fig.savefig('../out/a20_precipeffect_ktdist.png'); plt.close(fig)

fig,ax=plt.subplots(1,1)
ax.hist(nlyrout[precipout==0],bins=np.linspace(-0.5,10.5,12))
ax.set_xlabel('Number of cloud layers')
ax.set_ylabel('Counts');
plt.tight_layout(); 
fig.savefig('../out/a20_nlyrdist.png'); plt.close(fig)

fig,ax=plt.subplots(1,5)
for ilyr in range(0,5):
    fplt=np.logical_and(precipout==0,nlyrout==ilyr)
    ax[ilyr].hist2d(ktmeanout[fplt,0],ktstdout[fplt,0],bins=20)
    ax[ilyr].set_xlim([0,1.4])      
    ax[ilyr].set_ylim([0,0.5])    
    
plt.hist2d(szaout[ktout>0],ktout[ktout>0],bins=40)

fprecip=np.logical_and(np.logical_and(ktout>0,eleout>20),precipout==1)
plt.hist(ktout[fprecip],alpha=0.5)
fprecip=np.logical_and(np.logical_and(ktout>0,eleout>20),precipout==0)
plt.hist(ktout[fprecip],alpha=0.5)
    
