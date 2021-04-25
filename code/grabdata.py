#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for grabbing data from ARM THREDDS catalogues using siphon
@author: Mónica Zamora Z., UChile. http://mzamora.github.io
"""

from __future__ import print_function
from siphon.catalog import TDSCatalog
import pandas as pd

# i ordered data in 2 catalogues. Add xml link after logging in!
#cat=TDSCatalog('YourARMlink')
cat=TDSCatalog('https://archive.arm.gov/orders/catalog/orders/zamoram1/225099/catalog.xml?ticket=ST-4668-OlIM3xgaZeuGe2jEEfBeMlZGF0ssso')
dss=list(cat.datasets) # list of files in the dataset

# types of files (manually putting the different prefixes of the ordered products)
prfx1='corxsacrgridrhiM1.c1.'
prfx2='corkasacrgridrhiM1.c1.'

# set date (loopable)
yy=2019
mm=4
dd=26
thisday=pd.datetime(yy,mm,dd)
# beginning of the filename to search for
nm1=prfx1+thisday.strftime('%Y%m%d')
nm2=prfx2+thisday.strftime('%Y%m%d')
# filenames corresponding to that date
matches1 = [match for match in dss if nm1 in match]
matches2 = [match for match in dss if nm2 in match]

#access the variable list in the 1st set of nc files
ds=cat.datasets[matches1[0]] #for the first file
remoteds=ds.remote_access() #access the nc file remotely
remoteds #print its data
t=remoteds.variables['time'][:] #to grab data from the remote dataset

#access the variable list in the 2nd set of nc files
ds=cat.datasets[matches2[0]] #for the first file
remoteds=ds.remote_access() #access the nc file remotely
remoteds #print its data

# loop for grabbing or downloading
#for match in matches1:  
#    ds=cat.datasets[match]
    #ds.download() #this will download the nc file!
    #remoteds=ds.remote_access() #access the nc file remotely
    #t=remoteds.variables['time'][:]
    