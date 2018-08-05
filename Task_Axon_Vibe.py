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



