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


#==========
# data
#==========
CHICAGO_RECO_DATA = "some.tsv"
CHICAGO_RECO_HEADER = "some_header" 

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

        print("Result in km:", distance)



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

data = initDataWithHeader(CHICAGO_RECO_DATA, CHICAGO_RECO_HEADER)

# measure distance between query location and result location
idx = 1000
type(data['result_lat'][idx])
query_lat = data['query_lat'][idx]
query_lon = data['query_lon'][idx]
result_lat = data['result_lat'][idx]
result_lon = data['result_lon'][idx]

measureDistance(query_lat, query_lon, result_lat, result_lon)

# ToDo : how to plot distance distribution for each user
# get unique set of user id 
unique_user_id = data['user_int'].unique()
type(unique_user_id) # narray
num_unique_user = len(unique_user_id)
num_unique_user

# for each user, get all result place_int and measure the distance, plot it in chart
for u_idx in range(num_unique_user): 
    print 'user: ', u_idx
    userData = data[data['user_int'] == u_idx]
    user_queries = userData['query_raw'].unique() #ToDo : remove ' around query string
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
            num_sub_records = len(user_time_query_data)
        
        for r_idx in range(num_sub_records): 
               q_lat = user_time_query_data[user_time_query_data['result_index'] == r_idx]['query_lat']
               q_lat = q_lat.astype(float)
             
               q_lon = user_time_query_data[user_time_query_data['result_index'] == r_idx]['query_lon']
               q_lon = q_lon.astype(float)
               
               r_lat = user_time_query_data[user_time_query_data['result_index'] == r_idx]['result_lat']
               r_lat = r_lat.astype(float)
               
               r_lon = user_time_query_data[user_time_query_data['result_index'] == r_idx]['result_lon']
               r_lon = r_lon.astype(float)
               
               measureDistance(q_lat, q_lon, r_lat, r_lon)
        
        

        
        
        
