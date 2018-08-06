#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:54:26 2018

@author: soojunghong

@about : Axon Vibe task 

@date : August 2. 2018 
"""

#-------------------
# 0. Import libs 
#-------------------
import os 
import pandas as pd
import numpy as np 

from mapsplotlib import mapsplot as mplt
from gmplot import gmplot



#-------------------
#  1. Load data
#-------------------
CSV_PATH = "/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/"   

def load_data(filename, csv_path=CSV_PATH):
    file_path = os.path.join(csv_path, filename)
    return pd.read_csv(file_path)

location_data = load_data("locations.csv")
visits_data = load_data("visits.csv")


#------------------
# 2. Explore data
#------------------
location_data
location_data.head()
location_data.describe()
location_data.dtypes

type(location_data['recorded_timestamp'])
type(location_data['latitude'])


# latitude - horizontal direction 
# longitude - vertical direction 
# altitude - height from sea level 

sorted_location_w_id = location_data.sort_values(['id'], ascending=True)
sorted_location_w_id # 21211074 ~ 22358769

visits_data
visits_data.head()
visits_data.describe()
visits_data.dtypes

sorted_visits_w_id = visits_data.sort_values(['id'], ascending=True)
sorted_visits_w_id # 147967 ~ 170716


#--------------------------------
# data plotting with a few data
"""
def plotLocation(data): 
    lati = data["latitude"] 
    longi = data["longitude"] 

    center_lat = np.mean(lati)
    center_lon = np.mean(longi)
    zoom = 15

    #fileName = outfile+'.html'
    gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, zoom) 
    gmap.scatter(lati, longi)
    gmap.draw('plot.html')


few_loc = location_data.loc[0:10]
few_loc 
plotLocation(few_loc)
"""

#-------------------------------------
# plot position with my data on map
#-------------------------------------
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

gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, 18)#13)

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
gmap.scatter(lati, longi, '#228B22', size=8, marker=False)
                       
# Draw
gmap.draw("/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/test_all.html")


#------------------------
# partial data plotting 
part_lati = lati.loc[0:10]
part_longi = longi.loc[0:10]
center_part_lat = np.mean(part_lati)
center_part_lon = np.mean(part_longi)

gmap = gmplot.GoogleMapPlotter(center_part_lat, center_part_lon, 18)#13) -> this number for map zoom in 

gmap.scatter(part_lati, part_longi, '#00008B', size=8, marker=False)
                        
# Draw
gmap.draw("/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/test_partial.html")


#-----------------------
# visit data plotting

visits_data = load_data("visits.csv")

v_lati = visits_data["latitude"] 
v_longi = visits_data["longitude"] 


center_vlat = np.mean(v_lati)
center_vlon = np.mean(v_longi)

gmap = gmplot.GoogleMapPlotter(center_vlat, center_vlon, 18)#13)

# Scatter points  
gmap.scatter(v_lati, v_longi, '##8B008B', size=8, marker=False)
                       
# Draw
gmap.draw("/Users/soojunghong/Documents/2018 Job Applications/Axon_Vibe_Data_Scientist/Task/location_data/test_visit_all.html")


#------------------------------------------------------------
# How data (latitude or longitude) are distributed in time 
#------------------------------------------------------------


#---------------------------------------------------
# ToDo : location clustering with date and time
# make data point with condition with date or time and plot and see how it looks like 