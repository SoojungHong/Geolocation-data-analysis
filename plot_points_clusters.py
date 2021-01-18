import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN



"""
data location 
"""
file_path = "C:/Users/shong/PycharmProjects/sandbox_p2g-sensor-aggregation-and-conflation/rwo_parser/Ingolstadt_data/"
mc_path = "C:/Users/shong/PycharmProjects/sandbox_p2g-sensor-aggregation-and-conflation/rwo_parser/MicroCluster_csv/"

"""
plot scatter chart by feature name
"""


def plot_observation_by_feature(feature_name):
    feature_file = feature_name + "_merged.csv"
    df = pd.read_csv(file_path + feature_file)
    print('# of observation points : ', len(df))
    df.plot(kind="scatter", x="position::longitude_degrees", y="position::latitude_degrees", alpha=0.4,
            figsize=(20, 20), s=1, label=feature_name + " observation points")

    plt.legend()
    plt.show()


"""
plot cluster centroids in cluster result (in file)
"""


def plot_cluster_centroids_in_cluster_file(filename):
    df = pd.read_csv(filename)
    print('# of centroids : ', len(df))
    df.plot(kind="scatter", x="Lon", y="Lat", alpha=0.4, figsize=(20, 20), s=1, label="centroids (points) in clusters")

    plt.legend()
    plt.show()


"""
clean dataframe to remove NaN, infinity or a value too large for dtype('float64')
"""


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


"""
centroid calculation using mean of lat, mean of lon
"""


def _get_centermost_point(cluster):
    mean_point = (np.mean(cluster["position::latitude_degrees"]), np.mean(cluster["position::longitude_degrees"]))
    return tuple(mean_point)


"""
centroid calculation using sine and cosine
"""


def get_centermost_point(cluster):
    x = 0.0
    y = 0.0
    z = 0.0

    for i, clus in cluster.iterrows():
        latitude = math.radians(clus["position::latitude_degrees"])
        longitude = math.radians(clus["position::longitude_degrees"])
        x += math.cos(latitude) * math.cos(longitude)
        y += math.cos(latitude) * math.sin(longitude)
        z += math.sin(latitude)

    total = len(cluster)

    x = x / total
    y = y / total
    z = z / total

    central_longitude = math.atan2(y, x)
    central_square_root = math.sqrt(x * x + y * y)
    central_latitude = math.atan2(z, central_square_root)

    mean_location = {
        math.degrees(central_latitude),
        math.degrees(central_longitude)
    }

    return mean_location


def get_centermost_point2(cluster):
    lat = cluster["position::latitude_degrees"]
    lon = cluster["position::longitude_degrees"]
    mean_location = {
        math.degrees(sum(lat) / len(lat)),
        math.degrees(sum(lon) / len(lon))
    }

    return mean_location


"""
DBSCAN clustering
"""


def plot_clusters_with_DBSCAN_by_feature(feature):
    feature_file = feature + "_merged.csv"
    df = pd.read_csv(file_path + feature_file)
    geo_labels = ["position::latitude_degrees", "position::longitude_degrees"]
    new_df = clean_dataset(df[geo_labels])

    """
    calculate eps, min_sample and DBSCAN
    """
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    default_eps = 0.00001
    # db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(new_df))
    # db = DBSCAN(eps=epsilon, min_samples=2).fit(new_df)
    db = DBSCAN(eps=default_eps, min_samples=2).fit(new_df)

    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([new_df[cluster_labels == n] for n in range(num_clusters)])
    clusters = clusters[:-1]  # remove empty last dataframe

    """
    centroid of all clusters 
    """
    centermost_points = clusters.map(_get_centermost_point)
    lats, lons = zip(*centermost_points)
    # rep_points = pd.DataFrame({"position::longitude_degrees": lons, "position::latitude_degrees": lats})

    """
    plot clustering results and centroid in lat-lon 2D-space
    """
    fig, ax = plt.subplots(figsize=[10, 6])
    rs_scatter = ax.scatter(lons, lats, c='r', edgecolor='None', alpha=0.7, s=30)
    df_scatter = ax.scatter(new_df["position::longitude_degrees"], new_df["position::latitude_degrees"], c='b',
                            alpha=0.9, s=3)
    ax.set_title("DBSCAN Clustering on " + feature + " observations in Ingolstadt")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend([df_scatter, rs_scatter], [feature + " Observations", "Cluster centers"], loc='upper right')
    plt.show()


def plot_two_observations(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    fig, ax = plt.subplots(figsize=[10, 6])
    mc_scatter = ax.scatter(df1["position::longitude_degrees"], df1["position::latitude_degrees"], c='#33d7ff',
                            alpha=0.9, s=3)
    rs_scatter = ax.scatter(df2["position::longitude_degrees"], df2["position::latitude_degrees"], c='b',
                            edgecolor='None', alpha=0.9, s=4)
    ax.set_title("Plot observations of two observations")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.legend()
    plt.show()


"""
DBSCAN clustering
"""


def cluster_only_with_DBSCAN(feature):
    feature_file = feature + "_merged.csv"
    df = pd.read_csv(file_path + feature_file)
    geo_labels = ["position::latitude_degrees", "position::longitude_degrees"]

    new_df = clean_dataset(df[geo_labels])

    """
    calculate eps, min_sample and DBSCAN
    """
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    default_eps = 0.00001  # 0.002 (2m tolerance for lane boundary), signs and poles are 3m tolerance
    # db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(new_df))
    # db = DBSCAN(eps=epsilon, min_samples=2).fit(new_df)
    db = DBSCAN(eps=default_eps, min_samples=2).fit(new_df)

    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([new_df[cluster_labels == n] for n in range(num_clusters)])
    clusters = clusters[:-1]  # FIX: remove empty last dataframe
    print("number of clusters : ", num_clusters)

    """
    centroid of all clusters 
    """
    centermost_points = clusters.map(_get_centermost_point)
    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({"position::longitude_degrees": lons, "position::latitude_degrees": lats})

    """
    plot clustering results and centroid in lat-lon 2D-space
    """
    fig, ax = plt.subplots(figsize=[10, 6])
    rs_scatter = ax.scatter(lons, lats, c='b', edgecolor='None', alpha=0.7, s=1)
    ax.set_title("DBSCAN Clustering on " + feature + " observations in Ingolstadt")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend([rs_scatter], ["Cluster centroid"], loc='upper right')

    plt.show()


def read_microcluster_centroids(filename):
    df = pd.read_csv(mc_path + filename)
    X = []
    for i in range(len(df["Lon"])):
        elem = [df["Lon"][i], df["Lat"][i]]
        X.append(elem)

    return X


def plot_DBSCAN_Microcluster_by_feature(feature):
    if feature == "pole":
        mc_file = "MicroCluster_poles.csv"
    elif feature == "roadboundary":
        mc_file = "MicroCluster_road-boundary_only.csv"  # "MicroCluster_road-boundary.csv"

    dbs_centroids = cluster_only_with_DBSCAN(feature)
    # mc_centroids = read_microcluster_centroids(mc_file)
    mc_centroids = pd.read_csv(mc_path + mc_file)

    """
    plot both 
    """
    fig, ax = plt.subplots(figsize=[10, 6])
    mc_scatter = ax.scatter(mc_centroids["Lon"], mc_centroids["Lat"], c='#33d7ff', alpha=0.9, s=3)
    rs_scatter = ax.scatter(dbs_centroids["position::longitude_degrees"], dbs_centroids["position::latitude_degrees"],
                            c='b', alpha=0.9, s=4)
    ax.set_title("Clustering comparison on " + feature + " observations in Ingolstadt")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend([mc_scatter, rs_scatter], ["Micro clusters", "DBSCAN clusters"], loc='upper right')

    plt.show()



def plot_sign_data_3d(sign_df):
    labels = ["position::longitude_degrees", "position::latitude_degrees", "position::altitude_meters", "details::classification::gfr_group", "HadSign"]
    df = sign_df[labels]
    df["HadSign"] = pd.Categorical(df["HadSign"])
    df["sign_code"] = df["HadSign"].cat.codes
    clustering_labels = ["position::longitude_degrees", "position::latitude_degrees", "position::altitude_meters", "sign_code"]
    df_clean = clean_dataset(df[clustering_labels])
    df_clean["HadSign"] = df["HadSign"]

    """
    set plot size 
    """
    subplotsize = [12., 10.]
    figuresize = [15., 10.]

    left = 0.5 * (1. - subplotsize[0] / figuresize[0])
    right = 1. - left
    bottom = 0.5 * (1. - subplotsize[1] / figuresize[1])
    top = 1. - bottom
    fig = plt.figure(figsize=(figuresize[0], figuresize[1]))
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    ax = fig.add_subplot(111, projection='3d')

    """
    colors for each scatter plot
    """
    color_labels = df_clean["HadSign"].unique()
    rgb_values = sns.color_palette("Set1", n_colors=len(color_labels))
    color_map = dict(zip(color_labels, rgb_values))

    groups = df_clean.groupby("HadSign")
    for name, group in groups:
        ax.plot(group["position::longitude_degrees"], group["position::latitude_degrees"], group["position::altitude_meters"], marker='o', linestyle='',
                ms=1.5, label=name, color=color_map[name])

    ax.legend(fontsize=8, bbox_to_anchor=(1.1, 1))
    ax.set_title("Sign observation in Ingolstadt", fontsize=15)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)

    ax.set_xlabel('Longitude', fontsize=15)
    ax.set_ylabel('Latitude', fontsize=15)
    ax.set_zlabel('Altitude', fontsize=15)

    plt.show()


if __name__ == "__main__":
    """
    # examples 

    #plot by feature name ("roadboundary", "lanemarking", "roadboundary", "pole")
    plot_observation_by_feature("barrier")

    #plot the cluster centroids
    plot_cluster_centroids_in_cluster_file(mc_path + "MicroCluster_road-boundary_only.csv")

    #plot two observations
    eg_file1 = "C:/Users/shong/PycharmProjects/sandbox_p2g-sensor-aggregation-and-conflation/rwo_parser/Ingolstadt_data/lanemarking_merged.csv"
    eg_file2 = "C:/Users/shong/PycharmProjects/sandbox_p2g-sensor-aggregation-and-conflation/rwo_parser/Ingolstadt_data/roadboundary_merged.csv"
    plot_two_observations(eg_file1, eg_file2)

    # plot clusters by feature ("pole", "roadboundary", "pole", "pole")
    feature_type = "roadboundary"
    plot_clusters_with_DBSCAN_by_feature(feature_type)
    """

    # cluster_only_with_DBSCAN(feature_type)
    feature_type = "roadboundary"
    plot_DBSCAN_Microcluster_by_feature(feature_type)
