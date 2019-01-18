# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:08:11 2019

@author: shong

@about: recommendation outlier detection model
"""


#=====================
# import libraries
#=====================
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import os 
import matplotlib.mlab as mlab
import numpy as np

#==========
# data
#==========
CHICAGO_RECO_DATA = "C:/Users/shong/Documents/data/somedata.tsv"
CHICAGO_RECO_HEADER = #confidential 
RESULT_PLOT = "C:/Users/shong/Documents/data/plot/"

#=============
# functions 
#=============
def readDataWithoutHeader(dataset): 
    file_path = dataset
    data = pd.read_csv(file_path, delimiter='\t')
    return data
    
def readData(dataset, header):
    file_path = dataset
    headers = header
    data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)
    #data = pd.read_csv(file_path, delimiter='\t', names=headers)
    return data 

def initDataWithHeader(dataset, header): 
    data = readData(dataset, header)
    data['user_int'] = pd.factorize(data.cookie)[0]
    data['place_int'] = pd.factorize(data.result_name)[0]
    return data

def convertToReadableDate(unix_time):
    result_ms = pd.to_datetime(unix_time, unit='ms')
    str(result_ms)
    return result_ms

def isTwoColumnsValueSame(df, column1, column2): 
    is_same = df[column1].equals(df[column2])
    print is_same
    return is_same

def measureDistance(query_lat, query_lon, result_lat, result_lon):
    # approximate radius of earth in km
    R = 6373.0

    #print 'query_lat : ', query_lat.empty
    if(query_lat.empty != True): 
        q_lat = radians(query_lat) 
        q_lon = radians(query_lon) 
        r_lat = radians(result_lat) 
        r_lon = radians(result_lon) 

        dlon = r_lon - q_lon
        dlat = r_lat - q_lat

        a = sin(dlat / 2)**2 + cos(q_lat) * cos(r_lat) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        #print("Result in km:", distance)
    else: 
        print("query_lat is empty")
        distance = 0
        
    return distance


def visualizeHorizontalBar(places, distances, query, userId): 
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcdefaults()
    places_input = places 
    y_pos = np.arange(len(places_input))
    #fig = plt.figure()
    distances_input = distances 

    plt.barh(y_pos, distances_input, align='center', alpha=0.5)
    plt.yticks(y_pos, places_input)
    plt.xlabel('distance (km)')
    plt.ylabel('place name')
    currtitle = 'Distance of recommendation places in query ' + query
    plt.title(currtitle)
    plt.show()
   
    """
    ToDo : bugFix 
   # f1 = query+"_distance.png"
    path = os.path.join(RESULT_PLOT, query)
    filename = path + ".png"
    filename.decode('utf-8')
    print filename
    fig.savefig(filename) 
    """

def histogramAllDistances(all_distance): 
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10,10))
    #bins = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15, 20, 30, 75, 100]
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 60, 80, 100]
    hist, bins = np.histogram(all_distance[0:47800], bins=bins)
    width = 0.7 * (bins[5] - bins[4])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Distance (km)')
    plt.ylabel('Number of POI')
    
    plt.show()


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


def histogram(all_z_scores): 
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(7,7))
    #bins = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15, 20, 30, 75, 100]
    bins = [-3.5, -3.4, -3.2, -3.1, -3.0, -2.5, -2, -1.5, -1, -0.5, 0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    #bins = range(-4, 4, 1)
    hist, bins = np.histogram(all_z_scores[0:47800], bins=bins)
    width = 0.6 * (bins[5] - bins[4])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Z-score')
    plt.ylabel('Number of POI')
    
    plt.show()


def outliers_z_score(dp, thres):
    threshold = thres
    mean_y = np.mean(dp)
    stdev_y = np.std(dp)
    z_scores = [(datapoint - mean_y)/stdev_y for datapoint in dp]
    print z_scores
    #return np.where(np.abs(z_scores) > threshold)  
    return z_scores 


def outliers_modified_z_score(dp, thres):
    threshold = thres #3.5

    median_y = np.median(dp)
    median_absolute_deviation_y = np.median([np.abs(datapoint - median_y) for datapoint in dp])
    modified_z_scores = [0.6745 * (datapoint - median_y) / median_absolute_deviation_y for datapoint in dp]
    print modified_z_scores
    #return np.where(np.abs(modified_z_scores) > threshold)
    return modified_z_scores 


#================
# Experiment 
#================  

# load data
data = readData(CHICAGO_RECO_DATA, CHICAGO_RECO_HEADER) 
data.shape # (58400, 16) 
data['result_index'] 
data['result_rank']
data

isTwoColumnsValueSame(data, 'result_index', 'result_rank')
isTwoColumnsValueSame(data, 'query_string', 'query_normalized')
isTwoColumnsValueSame(data, 'query_raw', 'query_string')

# add columns for user id, place id 
data = initDataWithHeader(CHICAGO_RECO_DATA, CHICAGO_RECO_HEADER)
data


# get unique set of user id 
unique_user_id = data['user_int'].unique()
type(unique_user_id) # narray
num_unique_user = len(unique_user_id)
num_unique_user # 1362
all_distance = list()

# for each user, get all result place_int and measure the distance, plot it in chart
for u_idx in range(num_unique_user): 
    print 'user id: ', u_idx
    userData = data[data['user_int'] == u_idx]
    user_queries = userData['query_raw'].unique() 
    num_queries = len(user_queries)
    
    for q_idx in range(num_queries):
        print 'query index: ', q_idx
        user_query_data = userData[(userData['user_int'] == u_idx) & (userData['query_raw'] == user_queries[q_idx])]
        num_records = len(user_query_data)
        
        # set time index in given records 
        user_query_data['time_int'] = pd.factorize(user_query_data.timestamp)[0]
        num_times = len(user_query_data['time_int'].unique())
  
        for t_idx in range(num_times): 
            user_time_query_data = user_query_data[user_query_data['time_int'] == t_idx]
            user_time_query_data['distance'] = 0 # initialize as zero
            num_sub_records = len(user_time_query_data)
            #print user_time_query_data
        
            for r_idx in range(num_sub_records): 
                q_lat = user_time_query_data[user_time_query_data['result_index'] == r_idx]['query_lat']
                q_lat = q_lat.astype(float)
             
                q_lon = user_time_query_data[user_time_query_data['result_index'] == r_idx]['query_lon']
                q_lon = q_lon.astype(float)
               
                r_lat = user_time_query_data[user_time_query_data['result_index'] == r_idx]['result_lat']
                r_lat = r_lat.astype(float)
               
                r_lon = user_time_query_data[user_time_query_data['result_index'] == r_idx]['result_lon']
                r_lon = r_lon.astype(float)
               
                currentQuery = user_time_query_data[user_time_query_data['result_index'] == r_idx]['query_raw']
                #print 'current query: ', currentQuery
                dist = measureDistance(q_lat, q_lon, r_lat, r_lon)
                
                #set distance 
                user_time_query_data.loc[user_time_query_data['result_index'] == r_idx, ['distance']] = dist
                
            # for each sub data frame, show the plot with distances between query location and result location
            # user - location 
            result = ['user_int', 'distance', 'place_name', 'query_raw']
            # print user_time_query_data[result]
            out = user_time_query_data[result]
            places = out['place_name']
            distances = out['distance'] # this is array of distances
            all_distance.extend(distances)
            inputQuery = out['query_raw']
            query = inputQuery.iloc[0]
            userId = out['user_int'].iloc[0]
            
            # visualize with horizontal bar chart 
            # visualizeHorizontalBar(places, distances, query, userId)

        
len(all_distance) #58400
all_distance

# plot all distances in dataset
histogramAllDistances(all_distance)


# calculate z-score
from scipy import stats
stats.zscore(all_distance)
z_scores = outliers_z_score(all_distance, thres=3)
modified_z_score = outliers_modified_z_score(all_distance, thres=3.5)


# plot z-score in histogram
histogram(z_scores)
histogram(modified_z_score)
