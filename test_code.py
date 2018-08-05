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


#-------------------
#  1. Load data
CSV_PATH = "/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/"   

def load_data(filename, csv_path=CSV_PATH):
    file_path = os.path.join(csv_path, filename)
    return pd.read_csv(file_path)

location_data = load_data("locations.csv")
visits_data = load_data("visits.csv")


#-------------------
# Not that great
#-------------------

import pandas as pd 
#cities = pd.read_csv('my_csv.csv')
file_path = os.path.join(CSV_PATH, "locations.csv")
cities = pd.read_csv(file_path)


# extract data we are interested in
lat = cities['latitude'].values
lon = cities['longitude'].values
population = cities['altitude'].values  #cities['population_total'].values
area = cities['horizontal_accuracy'].values #cities['area_total_km2'].values

# draw map background
fig = plt.figure(figsize=(8,8))
#m = Basemap(projection = 'lcc', resolution='h', lat_0 = 37.5, lon_0 = -119, width=1E6, height=1.2E6)
m = Basemap(projection = 'lcc', resolution='h', lat_0 = 46, lon_0 = 8, width=1E6, height=1.2E6)

m.shadedrelief() 
m.drawcoastlines(color = 'gray')
m.drawcountries(color = 'gray')
m.drawstates(color = 'gray')

# scatter city data, with color reflecting population and size reflecting area
#m.scatter(lon, lat, latlon = True, c = np.log10(population), s = area, cmap = 'Reds', alpha=0.5)
m.scatter(lon, lat, latlon=True)
# create colorbar and legend
plt.colorbar(label = r'$s\log_{10}(\rm population})$')
plt.clim(3,7)

# make legend with dummy points
for a in [100, 300, 500]:
    plt.scatter([], [], c = 'k', alpha=0.5, s = a, label=str(a) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower left')

"""
# using google map
mplt.register_api_key('your_google_api_key_here')
mplt.density_plot(df['latitude'], df['longitude'])
"""



#-----------
# test 
# reference : https://github.com/wrobell/geotiler/blob/master/examples/ex-basemap.py
# https://wrobell.dcmod.org/geotiler/usage.html
#-----------
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import logging
logging.basicConfig(level=logging.DEBUG)

import geotiler

bbox = 11.78560, 46.48083, 11.79067, 46.48283

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111)

#
# download background map using OpenStreetMap
#
mm = geotiler.Map(extent=bbox, zoom=18)

img = geotiler.render_map(mm)

#
# create basemap
#
map = Basemap(
    llcrnrlon=bbox[0], llcrnrlat=bbox[1],
    urcrnrlon=bbox[2], urcrnrlat=bbox[3],
    projection='merc', ax=ax
)

map.imshow(img, interpolation='lanczos', origin='upper')

#
# plot custom points
#
x0, y0 = 11.78816, 46.48114 # http://www.openstreetmap.org/search?query=46.48114%2C11.78816
x1, y1 = 11.78771, 46.48165 # http://www.openstreetmap.org/search?query=46.48165%2C11.78771
x, y = map((x0, x1), (y0, y1))
ax.scatter(x, y, c='red', edgecolor='none', s=10, alpha=0.9)

plt.savefig('ex-basemap.pdf', bbox_inches='tight')
plt.close()
"""

from gmplot import gmplot

# Place map
gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13)

# Polygon
golden_gate_park_lats, golden_gate_park_lons = zip(*[
    (37.771269, -122.511015),
    (37.773495, -122.464830),
    (37.774797, -122.454538),
    (37.771988, -122.454018),
    (37.773646, -122.440979),
    (37.772742, -122.440797),
    (37.771096, -122.453889),
    (37.768669, -122.453518),
    (37.766227, -122.460213),
    (37.764028, -122.510347),
    (37.771269, -122.511015)
    ])
gmap.plot(golden_gate_park_lats, golden_gate_park_lons, 'cornflowerblue', edge_width=10)

# Scatter points
top_attraction_lats, top_attraction_lons = zip(*[
    (37.769901, -122.498331),
    (37.768645, -122.475328),
    (37.771478, -122.468677),
    (37.769867, -122.466102),
    (37.767187, -122.467496),
    (37.770104, -122.470436)
    ])
gmap.scatter(top_attraction_lats, top_attraction_lons, '#3B0B39', size=40, marker=False)

# Marker
hidden_gem_lat, hidden_gem_lon = 37.770776, -122.461689
gmap.marker(hidden_gem_lat, hidden_gem_lon, 'cornflowerblue')

# Draw
gmap.draw("my_map.html")