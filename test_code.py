#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:45:49 2018

@author: soojunghong
"""

#-------------------------------
# Geo mapping with matplotlib
#-------------------------------

DATASETS_URL = "https://github.com/ageron/handson-ml/raw/master/datasets"
import os
import tarfile
from six.moves import urllib

HOUSING_PATH = "datasets/housing"
HOUSING_URL = DATASETS_URL + "/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.exists(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()
housing.head()

import matplotlib.pyplot as plt
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)
#plt.show()


housing.plot(kind="scatter", x="longitude", y="latitude",
    s=housing['population']/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=0.4, figsize=(10,7),
)
plt.legend()
plt.show()

#------------------------
# works but not pretty
#------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import io
import pandas as pd

u = u"""latitude,longitude
42.357778,-71.059444
39.952222,-75.163889
25.787778,-80.224167
30.267222, -97.763889"""

u

# read in data to use for plotted points
buildingdf = pd.read_csv(io.StringIO(u), delimiter=",")
lat = buildingdf['latitude'].values
lon = buildingdf['longitude'].values
lat
lon
# determine range to print based on min, max lat and lon of the data
margin = 2 # buffer to add to the range
lat_min = min(lat) - margin
lat_max = max(lat) + margin
lon_min = min(lon) - margin
lon_max = max(lon) + margin

# create map using BASEMAP
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(lon_max-lon_min)/2,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white',lake_color='#46bcec')
                 # convert lat and lon to map projection coordinates
lons, lats = m(lon, lat)
# plot points as red dots
m.scatter(lons, lats, marker = 'o', color='r', zorder=5)
plt.show()


#------------------
# show globe
# https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
#------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

plt.figure(figsize=(11, 11))
#m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
m = Basemap(projection='ortho', resolution=None, lat_0=47, lon_0=8)


fig = plt.figure(figsize=(8, 8))
#m = Basemap(projection='lcc', resolution=None,
#            width=8E6, height=8E6, 
#            lat_0=45, lon_0=-100,)

m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6, 
            lat_0=47, lon_0=8,)

m.etopo(scale=0.5, alpha=0.5)

# Map (long, lat) to (x, y) for plotting
#x, y = m(-122.3, 47.6)
lon = 8.297589
lat = 47.050662
x, y = m(lon, lat)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' switzerland', fontsize=12)


#--------------------------
# ToDo : California city
#--------------------------
# https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
