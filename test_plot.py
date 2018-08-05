#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:39:52 2018

@author: soojunghong
"""

#-------------------------------
# This looks good map plotting
#-------------------------------

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
gmap.draw("/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/mytest1.html")

#--------------------
# test with my data
#--------------------
import os 
import pandas as pd
import numpy as np 

from mapsplotlib import mapsplot as mplt
from gmplot import gmplot

CSV_PATH = "/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/"   

def load_data(filename, csv_path=CSV_PATH):
    file_path = os.path.join(csv_path, filename)
    return pd.read_csv(file_path)

location_data = load_data("locations.csv")
visits_data = load_data("visits.csv")

lati = location_data["latitude"] 
longi = location_data["longitude"] 

center_lat = np.mean(lati)
center_lon = np.mean(longi)

gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, 13)

# Scatter points
"""
top_attraction_lats, top_attraction_lons = zip(*[
    (37.769901, -122.498331),
    (37.768645, -122.475328),
    (37.771478, -122.468677),
    (37.769867, -122.466102),
    (37.767187, -122.467496),
    (37.770104, -122.470436)
    ])
"""    
gmap.scatter(lati, longi, '#3B0B39', size=40, marker=False)

# Draw
gmap.draw("/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/test1.html")