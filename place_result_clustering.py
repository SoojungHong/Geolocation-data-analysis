# libraries

import collections
import re
import sys

from experiments.place_category import joined_with_place_info
from lib.data_utils import *
from lib.file_utils import *
from lib.geo_utils import *
from lib.visual_utils import *

from sklearn.decomposition import PCA
from matplotlib import pyplot

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt




def load_google_w2v_model():
    from gensim.models import KeyedVectors
    filename = 'C:/Users/shong/Downloads/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'
    # binary file 3.4GB
    model = KeyedVectors.load_word2vec_format(filename, binary=True)  # 43 seconds to load
    return model


def get_all_queries_contains(query, df):
    contain_res_df = df[df['query_raw'].str.contains(query)]
    queries_contains_restaurant = contain_res_df['query_raw']
    return queries_contains_restaurant.unique()


def get_long_queries_contains(query, df):
    all_queries = df['query_raw'].unique()
    for i in range(len(all_queries)):
        if all_queries[i] == query:
            print('yes')
            return query


def get_one_token_query_set(queries):
    one_token_set = list()
    for i in range(len(queries)):
        current_query = queries[i]  # narray

        if len(current_query.split()) is 1:
            # remove special character # ToDo : handle the special char but dataset contains special char at the moment
            # cleanStr = re.sub('\W+','', current_query)
            # print(cleanStr)
            one_token_set.append(current_query)

    return one_token_set


def get_word_mover_distance(w2v_model, doc1, doc2):
    dist = w2v_model.wmdistance(doc1, doc2)
    return dist


def get_category_names(pid):
    match = merged_data[merged_data['ppid'] == pid]
    if match.empty:
        return None
    else:
        return match['category_names'].iloc[0]  # list


def get_word_vector(model, word):
    print(word)
    word = re.sub('\W+', ' ', word)
    tokens = word.split()
    word_vec = ''
    if len(tokens) > 1:
        for w in range(len(tokens)):
            cleaned_word = re.sub('\W+', '', tokens[w])
            curr_vec = model.get_vector(cleaned_word)
            print(curr_vec)
    else:
        word = re.sub('\W+', '', word)
        word = word.lower()
        if word in model.vocab:
            word_vec = model.get_vector(word)
            print(word_vec)
    return word_vec


def get_food_type(pid):
    match = merged_data[merged_data['ppid'] == pid]
    if match.empty:
        print("empty")
        return None
    else:
        return match['food_type'].iloc[0]  # list


def get_closest_category(category_list, model, pname):
    min_dist = sys.maxsize
    closest_cat_name = ""
    if type(category_list) is type(None):
        print('NoneType')
    else:
        if len(category_list) > 0:
            for c in range(len(category_list)):
                curr_category = re.sub('\W+', ' ',  category_list[c])
                place = re.sub('\W+', '', pname)
                distance = model.wmdistance(pname, curr_category)
                if distance < min_dist:
                    min_dist = distance
                    closest_cat_name = category_list[c]
            #print(min_dist)
            #print(closest_cat_name)
            #get_word_vector(model, closest_cat_name)
    return (re.sub('\W+', ' ', closest_cat_name)).lower()


def get_closest_foodtype(food_type_list, model, pname):
    # ToDo : do check if the food_type set is empty - if it is empty, return empty, empty food type will be added as a adjective
    min_dist = sys.maxsize
    closest_food_name = ""
    if type(food_type_list) is type(None):
        print('NoneType')
    else:
        if len(food_type_list) > 0:
            for c in range(len(food_type_list)):
                curr_food_type = re.sub('\W+', '', food_type_list[c])
                place = re.sub('\W+', '', pname)
                distance = model.wmdistance(place, curr_food_type)
                #print(curr_food_type, distance)
                if distance < min_dist:
                    min_dist = distance
                    closest_food_name = food_type_list[c]
            #print(min_dist)
            #print(closest_food_name
            #get_word_vector(model, closest_food_name)
    return (re.sub('\W+', ' ', closest_food_name)).lower()


def compose_place_description_by_word2vec(category_name, food_type):
    place_desc = food_type + "" + category_name
    return place_desc


def represent_to_vector(model, desc):
    split_desc = desc.split()
    # remove special charactor
    #sum_all_vec = np.ndarray((300,), float)  # we know the word vector size is 300
    sum_all_vec = np.zeros((300,), dtype=float)

    for des_i in range(len(split_desc)):
        print('test here')
        print(split_desc[des_i])
        token = re.sub('\W+', '', split_desc[des_i])
        print(token)
        if token in model.vocab:
            desc_vec = model.get_vector(token)
            sum_all_vec = sum_all_vec + desc_vec

    return sum_all_vec


def pca_place_description(df):
    # visualize the word vector (description to one vector)
    import numpy as np
    from sklearn.decomposition import PCA
    from matplotlib import pyplot
    pca = PCA(n_components=2)  # number of components to keep
    result = pca.fit_transform((df['desc_vec'].values).tolist())
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    places = list(final_df['place_desc'].values)
    for i, word in enumerate(places):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=8)
    pyplot.show()


# data
RECO_HEADER = []
RECO_DATA = "C:/data.tsv"

# read data
df = read_file(RECO_DATA, RECO_HEADER)  # (209921, 18)

# get records contain query_raw == 'restaurant'
#ret = get_all_queries_contains("'aurora public library'", df)
ret_query = get_long_queries_contains("'fast food restaurant'", df) # "'korean restaurant'", df)  #"'best restaurant near me'"
#one_set = get_one_token_query_set(ret)
print(ret_query)

# restaurant_query = one_set[0] # query - 'restaurants'
#restaurant_query = one_set[0]  # query - 'restaurant'
#print('query')
#print(restaurant_query)

# get all recommended place records on query and same query location
all_records = df[df['query_raw'] == ret_query]
all_timestamp = all_records['timestamp'].unique()

one_query_records = all_records[all_records['timestamp'] == all_timestamp[0]]  # we check only one timestamp which execute one query
# print(one_query_records)  # shape : 22 * 18

one_query_records['distance'] = one_query_records.apply(lambda row: measure_distance(row.query_lat, row.query_lon, row.result_lat, row.result_lon), axis=1)
# measure_distance_in_km(query_lat, query_lon, result_lat, result_lon)
#print(one_query_records['distance'])

# get joined place data frame
merged_data = joined_with_place_info()

# add columns 'category_names' and 'food_type'
one_query_records['category_names'] = one_query_records.apply(lambda row: get_category_names(row.result_name), axis=1)
# print(one_query_records)  # 22 * 19

one_query_records['food_type'] = one_query_records.apply(lambda row: get_food_type(row.result_name), axis=1)
# print(one_query_records)  # 22 * 20

# google word2vec
model = load_google_w2v_model()
one_query_records['closest_category'] = one_query_records.apply(lambda row: get_closest_category(row.category_names, model, row.name_chosen), axis=1)
one_query_records['closest_food_type'] = one_query_records.apply(lambda row: get_closest_foodtype(row.food_type, model, row.name_chosen), axis=1)
one_query_records['place_desc'] = one_query_records.apply(lambda row: compose_place_description_by_word2vec(row.closest_category, row.closest_food_type), axis=1)

# Prepare data frame which does not contain empty category or empty food type
final_df = one_query_records[one_query_records['place_desc'] != '']

final_df['desc_vec'] = final_df.apply(lambda row: represent_to_vector(model, row.place_desc), axis=1)

labels = ['query_raw', 'name_chosen', 'place_desc', 'desc_vec', 'distance']
print(final_df[labels])

# dimension reduction using PCA
pca = PCA(n_components=2)  # number of components to keep
desc_list = (final_df['desc_vec'].values).tolist()
print('print desc_list')
print(len(desc_list))
print(desc_list)
origin_vec = represent_to_vector(model, final_df['query_raw'].iloc[0])
origin_name = final_df['query_raw'].iloc[0]
desc_list.append(origin_vec)
print(len(desc_list))
result = pca.fit_transform(desc_list)
fig = plt.figure()
ax = plt.axes(projection='3d')

z_distance = (final_df['distance'].values).tolist()
z_distance.append(0)
x_data = result[:, 0]
y_data = result[:, 1]
ax.scatter3D(x_data, y_data, z_distance, c=z_distance, marker='o')  #cmap='Greens')
places = list(final_df['place_desc'].values)
place_names = list(final_df['name_chosen'].values)
for i in range(len(result)):
    if i == (len(result)-1):
        ax.text(x_data[i], y_data[i], z_distance[i], origin_name, color='Red', size=9)
    else:
        ax.text(x_data[i], y_data[i], z_distance[i], place_names[i]+':'+places[i], size=7)

plt.show()


# k-means algorithm

