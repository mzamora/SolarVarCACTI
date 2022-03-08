#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:41:11 2021

@author: Mónica Zamora, DIMEC-UChile
         mzamora.github.io
"""

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

# Read data

file_dir= '../data/'
prfx=['corqcrad1longM1.c2','cormwrret1liljclouM1.c2',
      'cormfrsrcldod1minM1.c1','corceilM1.b1',
      'corarsclkazrbnd1kolliasM1.c1','cor30smplcmask1zwangM1.c1']
lss=listdir(file_dir)

# set dates
d0=datetime(2018,10,1) #official start date https://www.arm.gov/research/campaigns/amf2018cacti
d1=datetime(2019,4,30) #official end date
dates=np.arange(d0,d1,timedelta(days=1)).astype(datetime)

#data outputs
tout=np.array([],dtype=datetime)
ghiout=np.array([]); difout=np.array([]); dniout=np.array([])
ghicsout=np.array([]); difcsout=np.array([]); dnicsout=np.array([])
ktout=np.array([]); ktmeanout=np.empty([0,4]); ktstdout=np.empty([0,4])
eleout=np.array([]); szaout=np.array([])
zblyrout=np.empty([0,10]); ztlyrout=np.empty([0,10]); nlyrout=np.array([])
precipout=np.array([])
df1min=pd.DataFrame(); df5min=pd.DataFrame(); df15min=pd.DataFrame(); 
df30min=pd.DataFrame(); df60min=pd.DataFrame()

#retrieve data
for thisday in dates:
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
        continue
    
    # 0 is RAD data
    matches = [match for match in lss if nm[0] in match]
    ds=Dataset(file_dir+matches[0])
    tghi=ds['time'][:].data
    ghi=ds['BestEstimate_down_short_hemisp'][:].data
    dif=ds['down_short_diffuse_hemisp'][:].data
    dni=ds['short_direct_normal'][:].data
    
    # 1 is LWP data from microwave radiometers doi:10.1109/TGRS.2007.903 (not used)
    matches = [match for match in lss if nm[1] in match]
    ds=Dataset(file_dir+matches[0])
    tlwp=ds['time'][:].data
    lwp=ds['be_lwp'][:].data #in g/m2
    pwv=ds['be_pwv'][:].data #precipitable water wapor (cm)

    # 2 is COD data (not used)
    matches = [match for match in lss if nm[2] in match]
    ds=Dataset(file_dir+matches[0])
    tcod=ds['time'][:].data
    cod=ds['optical_depth_instantaneous'][:].data
    reff=ds['effective_radius_instantaneous'][:].data

    # 3 is ceilometer data (not used)
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
        
    kt=pd.DataFrame(kt) #needed to calculate the rolling statistics
    twindows=[5,15,30,60]
    ktmean=np.zeros([len(tghi),4])
    ktstd=np.zeros([len(tghi),4])
    
    # plot and calculate rolling mean and std for different time windows
#    fig1,ax1=plt.subplots(2,3,figsize=[12,6])
#    ax1[0][0].plot(tghi/3600,ghi,tghi/3600,ghics)
#    ax1[0][0].set_xlabel('UTC Time (h)'); ax1[0][0].set_ylabel('GHI (W/m$^2$)')
#    ax1[0][0].legend(['Obs.','Clear sky'],loc='upper left'); 
#    ax1[0][0].set_xlim([8,24]); ax1[0][0].set_ylim([0,1400])
#    ax1[0][1].plot(tghi[fel]/3600,kt[fel].values,[8,24],[1,1],'r--'); 
#    ax1[0][1].set_xlabel('UTC Time (h)'); ax1[0][1].set_ylabel('$k_t$'); ax1[0][1].set_xlim([8,24])
    for ip in range(len(twindows)):
        twindow=twindows[ip]
        ktmean[:,ip]=kt.rolling(twindow,center=True).mean().squeeze()
        ktstd[:,ip]=kt.rolling(twindow,center=True).std().squeeze()
#        ax1[0][2].plot(tghi[fel]/3600,ktstd[fel,ip]/ktmean[fel,ip])
#        ax1[1][ip].plot(ktmean[fel,ip],ktstd[fel,ip],'.')
#    ax1[0][2].set_xlabel('UTC Time (h)'); ax1[0][2].set_ylabel('$\sigma_{k_t} / \overline{k_t}$')
#    ax1[0][2].legend(['5 min','15 min','30 min'])
#    for ip in range(len(twindows)):
#        ax1[1][ip].set_ylabel('$\sigma_{k_t}$'); ax1[1][ip].set_xlabel('$\overline{k_t}$')
#        ax1[1][ip].set_title('$\Delta t=$ '+str(twindows[ip])+' min'); 
#        ax1[1][ip].set_xlim([0,1.3]); ax1[1][ip].set_ylim([0,0.5])
#    plt.suptitle(thisday.strftime('%Y/%m/%d')); plt.tight_layout()
#    fig1.savefig('../out/ghikt_'+thisday.strftime('%Y%m%d')+'.png'); plt.close(fig1)
    
    # number of layers
    inodata=np.where(zblyr[:,0]==-9999)[0] # fill missing data with nans
    nlyrs[inodata]=np.nan
#    fig1,ax1=plt.subplots(3,1,figsize=[9,5])
#    ax1[0].plot(tkazr/3600,nlyrs); #ax1[0].set_xlabel('UTC Time (h)')
#    ax1[0].set_ylabel('#Cloud layers'); ax1[0].set_xlim([8,24])
#    ax1[0].set_xticks([])
    
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
#            ax1[1].plot([tkazr[it]/3600,tkazr[it]/3600],[zbs[ib],zts[ib]],'k')
            if zbs[ib]==160:
                precipkazr[it]=True
#    ax1[1].set_ylabel('Cloud location (m)'); ax1[1].set_xlim([8,24]); #ax1[1].set_xlabel('UTC Time (h)')
    # add in red times when data were not available
#    ax1[1].set_xticks([])
#    ax1[1].plot(np.matlib.repmat(tkazr[inodata]/3600,2,1),np.matlib.repmat([0,15000],len(inodata),1).T,'r')
    
#    ax1[2].plot(tkazr/3600,precipkazr)
#    ax1[2].set_xlabel('UTC Time (h)'); ax1[2].set_ylabel('Precipitation?')
#    ax1[2].set_xlim([8,24])
#    ax1[2].set_yticks([0,1]); ax1[2].set_yticklabels(['No','Yes']); 
#    plt.suptitle(thisday.strftime('%Y/%m/%d')); plt.tight_layout()
#    fig1.savefig('../out/clouds_'+thisday.strftime('%Y%m%d')+'.png'); plt.close(fig1)

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
    
    # ramp calculations
    ghi1min=pd.DataFrame({'ghi':ghi,'dni':dni,'ele':solpos['elevation'].values,'ncld':nlyrs1min,
                        'zb1st':zblyr1min[:,0],'zt1st':ztlyr1min[:,0]},
                       index=np.array([thisday+timedelta(seconds=ti) for ti in tghi]))
    ghi1min['ghiramp']=ghi1min['ghi'].diff()
    ghi1min['dniramp']=ghi1min['dni'].diff()
    ghi5min=ghi1min.resample('5T').agg({'ghi':'mean','dni':'mean','ele':'last','ncld':'last',
                                        'zb1st':'last','zt1st':'last'})
    ghi5min['ghiramp']=ghi5min['ghi'].diff()/5
    ghi5min['dniramp']=ghi5min['dni'].diff()/5
    ghi15min=ghi1min.resample('15T').agg({'ghi':'mean','dni':'mean','ele':'last','ncld':'last',
                                        'zb1st':'last','zt1st':'last'})
    ghi15min['ghiramp']=ghi15min['ghi'].diff()/15
    ghi15min['dniramp']=ghi15min['dni'].diff()/15
    ghi30min=ghi1min.resample('30T').agg({'ghi':'mean','dni':'mean','ele':'last','ncld':'last',
                                        'zb1st':'last','zt1st':'last'})
    ghi30min['ghiramp']=ghi30min['ghi'].diff()/30
    ghi30min['dniramp']=ghi30min['dni'].diff()/30
    ghi60min=ghi1min.resample('60T').agg({'ghi':'mean','dni':'mean','ele':'last','ncld':'last',
                                        'zb1st':'last','zt1st':'last'})
    ghi60min['ghiramp']=ghi60min['ghi'].diff()/60
    ghi60min['dniramp']=ghi60min['dni'].diff()/60
    
    # append variables to continue loop. only export when elevation>20
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
    
    df1min=df1min.append(ghi1min[ghi1min['ele']>20])
    df5min=df5min.append(ghi5min[ghi5min['ele']>20])
    df15min=df15min.append(ghi15min[ghi15min['ele']>20])
    df30min=df30min.append(ghi30min[ghi30min['ele']>20])
    df60min=df60min.append(ghi60min[ghi60min['ele']>20])
    
'''
post processing
'''  

fig,ax=plt.subplots(1,1)
binss=np.linspace(0,2,21)
ax.hist(ktout,bins=binss,alpha=0.5)  
ax.hist(ktout[precipout==0],bins=binss,alpha=0.5) 
ax.hist(ktout[precipout==1],bins=binss,alpha=0.5)  
ax.set_xlabel('$k_t$')
ax.set_ylabel('Counts');
ax.legend(['All data','No Precip.','Precip.'])
plt.tight_layout(); 
fig.savefig('../out/a20_precipeffect_ktdist.png'); plt.close(fig)

fig,ax=plt.subplots(1,1)
dtplt=nlyrout[precipout==0]
ax.hist(dtplt,bins=np.linspace(-0.5,10.5,12),weights=np.ones(len(dtplt))/len(dtplt))
ax.set_xlabel('Number of cloud layers')
ax.set_ylabel('Frequency');
plt.tight_layout(); 
fig.savefig('../out/a20_nlyrdist.png'); plt.close(fig)
np.nanmax(nlyrout[precipout==0]) #max number of cloud layers

fclass=np.zeros([len(ktout),6],dtype=bool)
fbase=np.logical_and(precipout==0,nlyrout==1)
fclass[:,0]=np.logical_and(np.logical_and(zblyrout[:,0]<2000,ztlyrout[:,0]<2000),fbase)
fclass[:,1]=np.logical_and(np.logical_and(zblyrout[:,0]<2000,np.logical_and(ztlyrout[:,0]>=2000,ztlyrout[:,0]<6000)),fbase)
fclass[:,2]=np.logical_and(np.logical_and(zblyrout[:,0]<2000,ztlyrout[:,0]>=6000),fbase)
fclass[:,3]=np.logical_and(np.logical_and(np.logical_and(zblyrout[:,0]>=2000,zblyrout[:,0]<6000),ztlyrout[:,0]<6000),fbase)
fclass[:,4]=np.logical_and(np.logical_and(np.logical_and(zblyrout[:,0]>=2000,zblyrout[:,0]<6000),ztlyrout[:,0]>=6000),fbase)
fclass[:,5]=np.logical_and(zblyrout[:,0]>=6000,fbase)

# Fig. 1 paper
fig,ax=plt.subplots(1,4,figsize=[10,6])
# a) histogram of the number of cloud layers
dtplt=nlyrout[precipout==0]
ax[0].hist(dtplt,bins=np.linspace(-0.5,10.5,12),weights=np.ones(len(dtplt))/len(dtplt))
ax[0].set_xlabel('Number of cloud layers')
ax[0].set_ylabel('Frequency');
ax[0].set_xlim([-0.5,7.5])
ax[0].set_xticks([0,1,2,3,4,5,6,7])
# b) cloud classification
fplt=np.logical_and(precipout==0,nlyrout==1)
zbs=zblyrout[fplt,0]    
zts=ztlyrout[fplt,0]    
ax[1].plot(zbs/1000,zts/1000,'.',c='darkgray',alpha=0.2)
ax[1].set_xlabel('Cloud base height (km)')
ax[1].set_ylabel('Cloud top height (km)')
ax[1].plot([0,2],[2,2],'r--')
ax[1].plot([0,6],[6,6],'r--')
ax[1].plot([2,2],[0,17.5],'r--')
ax[1].plot([6,6],[0,17.5],'r--')
ax[1].set_xlim([0,14])
ax[1].set_ylim([0,17.5])
ax[1].text(1,1,'Low',size=14,ha='center',va='center')
ax[1].text(1,4,'Low\ntall',size=14,ha='center',va='center')
ax[1].text(1,10,'Low \ntaller',size=14,ha='center',va='center')
ax[1].text(4,5,'Mid',size=14,ha='center',va='center')
ax[1].text(4,10,'Mid \ntall',size=14,ha='center',va='center')
ax[1].text(8,10,'High',size=14,ha='center',va='center')
# c) Frequency of each cloud category
ncldtype=np.nansum(fclass,axis=0)
ax[2].bar(['Low','Low\ntall','Low\ntaller','Mid','Mid\ntall','High'],ncldtype/np.sum(ncldtype))
ax[2].set_ylabel('Frequency')
# d) Precipitation events
binss=np.linspace(0,2,21)
ax[3].hist(ktout,bins=binss,alpha=0.5)  
ax[3].hist(ktout[precipout==0],bins=binss,alpha=0.5) 
ax[3].hist(ktout[precipout==1],bins=binss,alpha=0.5)  
ax[3].set_xlabel('$k_t$')
ax[3].set_ylabel('Counts');
ax[3].legend(['All data','No Precip.','Precip.'])
ax[3].set_xlim([0,1.35])

ax[0].set_position([0.08,0.6, 0.25, 0.35])
ax[1].set_position([0.4,0.1, 0.58, 0.85])
ax[2].set_position([0.73,0.18, 0.24, 0.22])
ax[3].set_position([0.08,0.1, 0.25, 0.35])
plt.figtext(0.08,0.97, 'a)', size=14)
plt.figtext(0.4,0.97, 'b)', size=14)
plt.figtext(0.73,0.42, 'c)', size=14)
plt.figtext(0.08,0.47, 'd)', size=14)
fig.savefig('../out/Fig1.png'); plt.close(fig)

# Fig 2 in a different file
# main_fig22only.py


ktout_np=ktout[precipout==0]
nlyr_np=nlyrout[precipout==0]
fclass_np=fclass[precipout==0]
tout_np=tout[precipout==0]
touth=[t.hour+t.minute/60 for t in tout_np]

# Fig 3 - kt statistics per time, number of clouds, and cloud type
fig,ax=plt.subplots(1,7,figsize=[10,5])
im=ax[0].hist2d(ktout_np,touth,bins=40,cmap=plt.cm.Greys,norm=pltc.LogNorm())
ax[0].set_ylabel('UTC time (h)')
ax[0].set_xlabel('$k_t$')
cb=fig.colorbar(im[3],ax=ax[0],location='bottom'); 

for ip in range(1,3):
    pltvar=ktout_np[nlyr_np==ip]
    ax[1].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step')
pltvar=ktout_np[nlyr_np>3]
ax[1].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step')
ax[1].legend(['1 layer','2 layers','$\geq$3 layers'],prop={'size':8},loc='upper right')
ax[1].set_xlabel('$k_t$')
ax[1].set_ylabel('Frequency')

for ip in range(1,3):
    pltvar=np.array(touth)[nlyr_np==ip]
    ax[2].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step')
pltvar=np.array(touth)[nlyr_np>3]
ax[2].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step')
ax[2].legend(['1 layer','2 layers','$\geq$3 layers'],prop={'size':8},loc='upper left')
ax[2].set_xlabel('UTC Time (h)')
ax[2].set_ylabel('Frequency')

ecs=['green','green','green','red','red','blue']
lss=['-','--',':','-','--','-']
for ip in range(0,3):
    pltvar=ktout_np[fclass_np[:,ip]]
    ax[3].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step',ec=ecs[ip],ls=lss[ip])
for ip in range(3,6):
    pltvar=ktout_np[fclass_np[:,ip]]
    ax[4].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step',ec=ecs[ip],ls=lss[ip])
ax[3].legend(['Low','Low tall','Low taller'],prop={'size':8},loc='upper right')
ax[4].legend(['Mid','Mid tall','High'],prop={'size':8},loc='upper left')
ax[3].set_xlabel('$k_t$'); ax[3].set_ylabel('Frequency')
ax[4].set_xlabel('$k_t$'); ax[4].set_ylabel('Frequency')

for ip in range(0,3):
    pltvar=np.array(touth)[fclass_np[:,ip]]
    ax[5].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step',ec=ecs[ip],ls=lss[ip])
for ip in range(3,6):
    pltvar=np.array(touth)[fclass_np[:,ip]]
    ax[6].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step',ec=ecs[ip],ls=lss[ip])
ax[5].legend(['Low','Low tall','Low taller'],prop={'size':8},loc='upper left')
ax[6].legend(['Mid','Mid tall','High'],prop={'size':8},loc='upper right')
ax[5].set_xlabel('UTC Time (h)'); ax[6].set_xlabel('UTC Time (h)')
ax[5].set_ylabel('Frequency'); ax[6].set_ylabel('Frequency')

for ip in [1,3,4]:
    ax[ip].set_ylim([0,0.4])
for ip in [2,5,6]:
    ax[ip].set_ylim([0,0.2])
    ax[ip].set_xticks([11,13,15,17,19,21])
for ip in [3,4,5,6]:
    ax[ip].set_yticks([])
    ax[ip].set_ylabel('')
ax[0].set_position([0.05,0.3,0.2,0.6])
cb.ax.set_position([0.05,0.1, 0.2, 0.1])
ax[1].set_position([0.32,0.58, 0.22, 0.38])
ax[2].set_position([0.32,0.09, 0.22, 0.38])
ax[3].set_position([0.545,0.58, 0.22, 0.38])
ax[5].set_position([0.545,0.09,0.22, 0.38])
ax[4].set_position([0.77,0.58, 0.22, 0.38])
ax[6].set_position([0.77,0.09, 0.22, 0.38])
cb.ax.set_xlabel('Counts')
plt.figtext(0.05,0.91, 'a)', size=14)
plt.figtext(0.32,0.97, 'b)', size=14)
plt.figtext(0.545,0.97, 'c)', size=14)
plt.figtext(0.77,0.97, 'd)', size=14)
plt.figtext(0.32,0.48, 'e)', size=14)
plt.figtext(0.545,0.48, 'f)', size=14)
plt.figtext(0.77,0.48, 'g)', size=14)
fig.savefig('../out/Fig3_cloudkt.png'); plt.close(fig)



fig,ax=plt.subplots(1,4,figsize=[14,4])
for iwindow in range(4):    
    for ilyr in range(7):
        fplt=np.logical_and(precipout==0,nlyrout==ilyr)
        ax[iwindow].plot(ktmeanout[fplt,iwindow],ktstdout[fplt,iwindow],'.',label=str(ilyr)+' layers')
    ax[iwindow].set_xlim([0,1.5])
    ax[iwindow].set_ylim([0,0.5])
    ax[iwindow].set_xlabel('$\overline{k_t}$')
    ax[iwindow].set_title('$\Delta t= '+str(twindows[iwindow])+'$ min')
ax[2].legend(loc='upper right')
ax[0].set_ylabel('$\sigma_{k_t}$')
plt.tight_layout()
fig.savefig('../out/a20_ktmeanstd_points.png'); plt.close(fig)


# 2d histograms of kt vs elevations and SZA
fig,ax=plt.subplots(1,2,figsize=[8,3])
im=ax[0].hist2d(eleout,ktout,bins=40,cmap=plt.cm.Greys,norm=pltc.LogNorm())
ax[0].set_xlabel('Elevation Angle (º)')
ax[0].set_ylabel('$k_t$')
cb=fig.colorbar(im[3],ax=ax[0]); cb.ax.set_ylabel('Counts')
im=ax[1].hist2d(szaout,ktout,bins=40,cmap=plt.cm.Greys,norm=pltc.LogNorm())
ax[1].set_xlabel('Solar Zenith Angle (º)')
ax[1].set_ylabel('$k_t$')
cb=fig.colorbar(im[3],ax=ax[1]); cb.ax.set_ylabel('Counts')
plt.tight_layout()
fig.savefig('../out/a20_ktvselevsza.png'); plt.close(fig)

# Fig 4 - effect of dt in kt vs sigma
fig,ax=plt.subplots(1,4,figsize=[11,2.5])
for iwindow in range(4):
        fplt=precipout==0
        #ax[iwindow].plot(ktmeanout[fplt,iwindow],ktstdout[fplt,iwindow],'.',alpha=0.005)
        im=ax[iwindow].hist2d(ktmeanout[fplt,iwindow],ktstdout[fplt,iwindow],
                                   bins=40,cmap=plt.cm.Greys,norm=pltc.LogNorm(vmin=1,vmax=2e4))
        ax[iwindow].set_xlim([0,1.3])
        ax[iwindow].set_ylim([0,0.45])
        ax[iwindow].set_xlabel('$\overline{k}_{t}$')
        ax[iwindow].set_title('$\Delta t= '+str(twindows[iwindow])+'$ min')
ax[0].set_ylabel('$\sigma_{k_{t}}$')
for ip in range(1,4):
    ax[ip].set_yticks([])
cbx = fig.add_axes([0.92, 0.2, 0.02, 0.7])
cb=fig.colorbar(im[3], cax=cbx)
cb.ax.set_ylabel('Counts')
plt.subplots_adjust(left=0.07, right=.91, wspace=0.05, hspace=0.01, bottom=0.2, top=0.9)
fig.savefig('../out/Fig4_ktsd_perdt.png');

# Fig 5 - dotted version

fig,ax=plt.subplots(6,4,figsize=[9,8])
for icld in range(6):
    for iwindow in range(4):
        ns=len(ktmeanout[fclass[:,icld],iwindow])
        ws=np.ones(ns)/ns
        im=ax[icld][iwindow].hist2d(ktmeanout[fclass[:,icld],iwindow],
                              ktstdout[fclass[:,icld],iwindow],
                                   bins=40,
                                   cmap=plt.cm.binary,
                                   weights=ws,
                                   #norm=pltc.LogNorm(vmax=0.05),
                                   #norm=pltc.Normalize()
                                   )
        if iwindow==3:
            cb=fig.colorbar(im[3], ax=ax[icld][iwindow])
            cb.ax.set_ylabel('Frequency')
            plt.show()
        ax[icld][iwindow].set_xlim([0,1.3])
        ax[icld][iwindow].set_ylim([0,0.45])

for iwindow in range(4):
    ax[5][iwindow].set_xlabel('$\overline{k}_{t}$')
    ax[0][iwindow].set_title('$\Delta t= '+str(twindows[iwindow])+'$ min')
for icld in range(6):
    ax[icld][0].set_ylabel('$\sigma_{k_t}$')
for iwindow in range(1,4):
    for icld in range(6):
        ax[icld][iwindow].set_yticks([])
for iwindow in range(4):
    for icld in range(5):
        ax[icld][iwindow].set_xticks([])
plt.subplots_adjust(left=0.07, right=.9, wspace=0.05, hspace=0.1,
                    bottom=0.08, top=0.95)
labels=['Low','Low tall','Low taller','Mid','Mid tall','High']
for iw in range(4):
    for icld in range(6):
        ax[icld][iw].text(0.1,0.36,labels[icld])
        
wd=0.19; ht=0.137        
for iw in range(4):
    for ic in range(6):
        ax[ic][iw].set_position([0.07+iw*(0.01+wd),0.813-ic*(0.01+ht),wd,ht])
fig.savefig('../out/Fig5_ktnstd_percloudtype.png'); plt.close(fig)

# Fig 5 - pdf line version
ecs=['brown','red','tomato','lightsalmon']
alphas=[0.8,0.7,0.6,0.5]

fig,ax=plt.subplots(6,2,figsize=[4,8])
labels=['Low','Low tall','Low taller','Mid','Mid tall','High']
for iclass in range(6):
    pltvar=ktout[fclass[:,iclass]]
    ax[iclass][0].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step',ec='black',lw=1.3)
    ax[iclass][0].set_xlim([0,1.3])
for iwindow in range(4):
    for iclass in range(6):
        pltvar=ktmeanout[fclass[:,iclass],iwindow]
        ax[iclass][0].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),histtype=u'step',
                           ec=ecs[iwindow],alpha=alphas[iwindow],lw=1.3)
for iclass in range(6):
    ax[iclass][1].set_xlim([0,0.5])
for iwindow in range(4):
    for iclass in range(6):
        pltvar=ktstdout[fclass[:,iclass],iwindow]
        ax[iclass][1].hist(pltvar,weights=np.ones(len(pltvar))/len(pltvar),
                           histtype=u'step',ec=ecs[iwindow],alpha=alphas[iwindow],lw=1.3)

ax[5][1].set_xlabel('$\sigma_{k_{t,\Delta t}}$')
ax[5][0].set_xlabel('$\overline{k}_{t,\Delta t}$')
for ip in range(6): ax[ip][0].set_ylabel('Frequency')
for ip in range(5): 
    for ip2 in range(2): ax[ip][ip2].set_xticks([])
ax[1][1].set_yticks([0,0.3,0.6])
ax[2][1].set_yticks([0,0.3,0.6])
ax[3][1].set_yticks([0,0.2])
ax[4][1].set_yticks([0,0.2,0.4])
plt.subplots_adjust(left=0.14, right=.99, wspace=0.25, hspace=0.12, 
                    bottom=0.07, top=0.94)
ax[0][0].text(1,0.21,'Low')
ax[0][1].text(0.39,0.46,'Low')
ax[1][0].text(0.78,0.25,'Low tall')
ax[1][1].text(0.3,0.55,'Low tall')
ax[2][0].text(0.66,0.235,'Low taller')
ax[2][1].text(0.26,0.72,'Low taller')
ax[3][0].text(0.09,0.187,'Mid')
ax[3][1].text(0.4,0.32,'Mid')
ax[4][0].text(0.8,0.171,'Mid tall')
ax[4][1].text(0.31,0.5,'Mid tall')
ax[5][0].text(0.09,0.36,'High')
ax[5][1].text(0.37,0.49,'High')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='black', lw=1.3),
                Line2D([0], [0], color='brown', alpha=0.8, lw=1.3),
                Line2D([0], [0], color='red', alpha=0.7, lw=1.3),
                Line2D([0], [0], color='tomato', alpha=0.6, lw=1.3),
                Line2D([0], [0], color='lightsalmon', alpha=0.5, lw=1.3)]
ax[0][0].legend(custom_lines,['1 min','5 min','15 min','30 min','60 min'],loc='upper center', 
                bbox_to_anchor=(1, 1.45), ncol=3,prop={'size':8})


# Fig 6 ramp statistics

fclass[:,0]

#### ramps ####
fig,ax=plt.subplots(1,7,figsize=(9,3))
ecs=['black','brown','red','tomato','lightsalmon']
alphas=[1,0.8,0.7,0.6,0.5]

# ramp CDF per cloud type and dt
def plotrampcdf(df,axx,ist):
    count,bincount = np.histogram(df['ghiramp'], bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    axx.plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
def plotrampcdfclds(df,ax,ist):
    zbs=df['zb1st'].values
    zts=df['zt1st'].values
    nclds=df['ncld'].values
    print(np.sum(nclds==0))
    fbase=nclds==1 #np.logical_and(precipout==0,nlyrout==1)
    flow=np.logical_and(np.logical_and(zbs<2000,zts<2000),fbase)
    flowtall=np.logical_and(np.logical_and(zbs<2000,np.logical_and(zts>=2000,zts<6000)),fbase)
    flowtaller=np.logical_and(np.logical_and(zbs<2000,zts>=6000),fbase)
    fmid=np.logical_and(np.logical_and(np.logical_and(zbs>=2000,zbs<6000),zts<6000),fbase)
    fmidtall=np.logical_and(np.logical_and(np.logical_and(zbs>=2000,zbs<6000),zts>=6000),fbase)
    fhigh=np.logical_and(zbs>=6000,fbase)
    count,bincount = np.histogram(df['ghiramp'][flow].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[1].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['ghiramp'][flowtall].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[2].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['ghiramp'][flowtaller].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[3].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['ghiramp'][fmid].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[4].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['ghiramp'][fmidtall].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[5].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['ghiramp'][fhigh].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[6].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)

# all cases - ghi
plotrampcdf(df1min,ax[0],0)
plotrampcdf(df5min,ax[0],1)
plotrampcdf(df15min,ax[0],2)
plotrampcdf(df30min,ax[0],3)
plotrampcdf(df60min,ax[0],4)
# per cloud type - ghi
plotrampcdfclds(df1min,ax,0)
plotrampcdfclds(df5min,ax,1)
plotrampcdfclds(df15min,ax,2)
plotrampcdfclds(df30min,ax,3)
plotrampcdfclds(df60min,ax,4)

ax[0].set_position([0.75,0.59,0.22,0.39])
ax[1].set_position([0.06,0.59, 0.22, 0.39])
ax[2].set_position([0.29,0.59, 0.22, 0.39])
ax[3].set_position([0.52,0.59, 0.22, 0.39])
ax[4].set_position([0.06,0.16,0.22, 0.39])
ax[5].set_position([0.29,0.16, 0.22, 0.39])
ax[6].set_position([0.52,0.16, 0.22, 0.39])
ax[1].set_ylabel('Cumulative frequency',size=8)
ax[4].set_ylabel('Cumulative frequency',size=8)
for axi in ax:  
    axi.set_ylim([0,1])
    axi.set_xlim([-300,300])
pltlabels=['All','Low','Low tall','Low taller','Mid','Mid taller','High']
for ip in range(7):
    ax[ip].text(-280,0.85,pltlabels[ip])
ax[0].legend(['1 min','5 min','15 min','30 min','60 min'],loc='lower center', 
             bbox_to_anchor=(0.5, -1.1),prop={'size':9},ncol=2)
for ip in [0,4,5,6]:
    ax[ip].set_xlabel('GHI ramp (W/m$^2$/min)')
for ip in [1,2,3]: ax[ip].set_xticks([])
for ip in [0,2,3,5,6]: ax[ip].set_yticks([])
ax[1].set_yticks([0,0.5,1])
ax[4].set_yticks([0,0.5,1])
fig.savefig('../out/Fig6_ghirampscdf.png'); plt.close(fig)
    

# DNI ramps
fig,ax=plt.subplots(1,7,figsize=(9,3))

# ramp CDF per cloud type and dt
def plotrampcdfDNI(df,axx,ist):
    count,bincount = np.histogram(df['dniramp'], bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    axx.plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
def plotrampcdfcldsDNI(df,ax,ist):
    zbs=df['zb1st'].values
    zts=df['zt1st'].values
    nclds=df['ncld'].values
    print(np.sum(nclds==0))
    fbase=nclds==1 #np.logical_and(precipout==0,nlyrout==1)
    flow=np.logical_and(np.logical_and(zbs<2000,zts<2000),fbase)
    flowtall=np.logical_and(np.logical_and(zbs<2000,np.logical_and(zts>=2000,zts<6000)),fbase)
    flowtaller=np.logical_and(np.logical_and(zbs<2000,zts>=6000),fbase)
    fmid=np.logical_and(np.logical_and(np.logical_and(zbs>=2000,zbs<6000),zts<6000),fbase)
    fmidtall=np.logical_and(np.logical_and(np.logical_and(zbs>=2000,zbs<6000),zts>=6000),fbase)
    fhigh=np.logical_and(zbs>=6000,fbase)
    count,bincount = np.histogram(df['dniramp'][flow].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[1].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['dniramp'][flowtall].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[2].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['dniramp'][flowtaller].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[3].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['dniramp'][fmid].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[4].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['dniramp'][fmidtall].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[5].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)
    count,bincount = np.histogram(df['dniramp'][fhigh].values, bins=100)
    pdf = count/sum(count); cdf = np.cumsum(pdf)
    ax[6].plot(bincount[1:], cdf,color=ecs[ist],alpha=alphas[ist],lw=1.3)

# all cases - DNI
plotrampcdf(df1min,ax[0],0)
plotrampcdf(df5min,ax[0],1)
plotrampcdf(df15min,ax[0],2)
plotrampcdf(df30min,ax[0],3)
plotrampcdf(df60min,ax[0],4)
# per cloud type - ghi
plotrampcdfclds(df1min,ax,0)
plotrampcdfclds(df5min,ax,1)
plotrampcdfclds(df15min,ax,2)
plotrampcdfclds(df30min,ax,3)
plotrampcdfclds(df60min,ax,4)

ax[0].set_position([0.75,0.59,0.22,0.39])
ax[1].set_position([0.06,0.59, 0.22, 0.39])
ax[2].set_position([0.29,0.59, 0.22, 0.39])
ax[3].set_position([0.52,0.59, 0.22, 0.39])
ax[4].set_position([0.06,0.16,0.22, 0.39])
ax[5].set_position([0.29,0.16, 0.22, 0.39])
ax[6].set_position([0.52,0.16, 0.22, 0.39])
ax[1].set_ylabel('Cumulative frequency',size=8)
ax[4].set_ylabel('Cumulative frequency',size=8)
for axi in ax:  
    axi.set_ylim([0,1])
    axi.set_xlim([-300,300])
pltlabels=['All','Low','Low tall','Low taller','Mid','Mid taller','High']
for ip in range(7):
    ax[ip].text(-280,0.85,pltlabels[ip])
ax[0].legend(['1 min','5 min','15 min','30 min','60 min'],loc='lower center', 
             bbox_to_anchor=(0.5, -1.1),prop={'size':9},ncol=2)
for ip in [0,4,5,6]:
    ax[ip].set_xlabel('DNI ramp (W/m$^2$/min)')
for ip in [1,2,3]: ax[ip].set_xticks([])
for ip in [0,2,3,5,6]: ax[ip].set_yticks([])
ax[1].set_yticks([0,0.5,1])
ax[4].set_yticks([0,0.5,1])
fig.savefig('../out/Fig6_dnirampscdf.png'); plt.close(fig)

# ramp statistics
ghirampstats=np.zeros((5,7,5))
#(1: dt, 2: cld type(0 is all), 3:percentile)

def statsrampcdf(df):
    estats=np.zeros((7,5))
    pctiles=[10,25,50,75,90]
    zbs=df['zb1st'].values
    zts=df['zt1st'].values
    nclds=df['ncld'].values
    fbase=nclds==1
    flow=np.logical_and(np.logical_and(zbs<2000,zts<2000),fbase)
    flowtall=np.logical_and(np.logical_and(zbs<2000,np.logical_and(zts>=2000,zts<6000)),fbase)
    flowtaller=np.logical_and(np.logical_and(zbs<2000,zts>=6000),fbase)
    fmid=np.logical_and(np.logical_and(np.logical_and(zbs>=2000,zbs<6000),zts<6000),fbase)
    fmidtall=np.logical_and(np.logical_and(np.logical_and(zbs>=2000,zbs<6000),zts>=6000),fbase)
    fhigh=np.logical_and(zbs>=6000,fbase)
    estats[0,:]=np.percentile(df['dniramp'], pctiles)
    estats[1,:]=np.percentile(df['dniramp'][flow].values,pctiles)
    estats[2,:]=np.percentile(df['dniramp'][flowtall].values,pctiles)
    estats[3,:]=np.percentile(df['dniramp'][flowtaller].values,pctiles)
    estats[4,:]=np.percentile(df['dniramp'][fmid].values,pctiles)
    estats[5,:]=np.percentile(df['dniramp'][fmidtall].values,pctiles)
    estats[6,:]=np.percentile(df['dniramp'][fhigh].values,pctiles)   
    return estats
    
# get statistics for each time interval 
ghirampstats[0,:,:]=statsrampcdf(df1min)
ghirampstats[1,:,:]=statsrampcdf(df5min)
ghirampstats[2,:,:]=statsrampcdf(df15min)
ghirampstats[3,:,:]=statsrampcdf(df30min)
ghirampstats[4,:,:]=statsrampcdf(df60min)


###### plotting the statistics
ecss=['black','green','green','green','red','red','blue']
stss=['-','-','--',':']
fig,ax=plt.subplots(1,2)
#p5s
for icld in range(7):
    ax[0].plot(ghirampstats[:,icld,0]) #,stss[icld],color=ecss[icld],lw=1.3)
#p95s
for icld in range(7):
    ax[1].plot(ghirampstats[:,icld,3],'.-',color=ecss[icld],alpha=alphass[icld],lw=1.3)
ax[0].set_xticks([0,1,2,3,4])
ax[0].set_xticklab
    
    
  
