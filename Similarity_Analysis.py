import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

similarity_df = pd.read_csv('C:/Users/joon6/OneDrive/Desktop/Covid Project/Python/SimilarityTable.csv')
country_dropped = similarity_df.drop(['location', 'b_location', 'c_location'], axis=1)
country_dropped = country_dropped.dropna()
scaled_df = StandardScaler().fit_transform(country_dropped)

# initialize kmeans parameters
kmeans_initialize = {
    "init": "random",
    "n_init": 10,
    "random_state": 1,
}

# create list to hold SSE values for each k
sse = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, **kmeans_initialize)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

# visualize results
plt.plot(range(1, 15), sse)
plt.xticks(range(1, 15))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
# Bend at k = 4, so 4 clusters are the most optimal


# using optimal number of clusters
kmeans = KMeans(init="random", n_clusters=4, n_init=10, random_state=1)

# fit k-means algorithm to data
kmeans.fit(scaled_df)

# append cluster assingments to original DataFrame
similarity_df['cluster'] = kmeans.labels_

# view updated DataFrame
print(similarity_df)

# view cluster for Singapore
print(similarity_df.loc[similarity_df['location'] == 'Singapore'])
print(similarity_df.loc[similarity_df['location'] == 'Singapore', 'cluster'])

# list countries in cluster 1
c1_countries = (similarity_df.loc[similarity_df['cluster'] == 1, 'location']).tolist()
print(c1_countries)
print(len(c1_countries))

# There are 59 countries in cluster 1.
# We only need a few countries that are similar to Singapore
# rather than an optimization for all countries, so lets divide into more clusters.

# using  of clusters
kmeans = KMeans(init="random", n_clusters=10, n_init=10, random_state=1)

# fit k-means algorithm to data
kmeans.fit(scaled_df)

# append cluster assignments to original DataFrame
similarity_df['cluster'] = kmeans.labels_

# view updated DataFrame
print(similarity_df.to_string())

# view cluster for Singapore
print(similarity_df.loc[similarity_df['location'] == 'Singapore', 'cluster'])
# Singapore is assigned to cluster 1


# list countries in cluster 1
c1_countries = (similarity_df.loc[similarity_df['cluster'] == 1, 'location']).tolist()
print(c1_countries)
# Similar countries: ['Hong Kong', 'Luxembourg', 'Qatar', 'Singapore']
