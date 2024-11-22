from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to run KMeans clustering with n clusters on cleaned data
def train_kmeans(cleaned_data, n_clusters,):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(cleaned_data)
    return kmeans

def inference_kmeans(cleaned_data, kmeans):
    kmeans.predict(cleaned_data)
    return kmeans

def clean_data(game_data_with_gvi):
    game_data = pd.read_csv(game_data_with_gvi)
    cleaned_data = game_data.dropna(subset=['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI'])
    # Get rid of infinite values
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0.0)
    return cleaned_data[['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']]

def clean_data_with_names(game_data_with_gvi):
    game_data = pd.read_csv(game_data_with_gvi)
    cleaned_data = game_data.dropna(subset=['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI'])
    # Get rid of infinite values
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0.0)
    return cleaned_data[['game_name', 'duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']]

# Elbow Method to determine optimal number of clusters
def elbow_method(cleaned_data, max_clusters):
    distortions = []
    for i in range(2, max_clusters):
        print(i)
        kmeans = train_kmeans(cleaned_data, i)
        distortions.append(kmeans.inertia_)
    # Identify the optimal number of clusters
    return distortions

# Function to get the cluster labels for each game
def get_cluster_labels(cleaned_data, kmeans):
    kmeans = inference_kmeans(cleaned_data, kmeans)
    return kmeans.labels_

# Plot the elbow method
def plot_elbow(distortions, max_clusters):
    plt.plot(range(2, max_clusters), distortions)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

# Function to append cluster labels to game data
def append_cluster_labels(game_data_with_gvi, n_clusters):
    cleaned_data = clean_data(game_data_with_gvi)
    kmeans = train_kmeans(cleaned_data, n_clusters)
    cluster_labels = get_cluster_labels(cleaned_data, kmeans)
    # Append cluster labels to cleaned_data
    cleaned_data_with_names = clean_data_with_names(game_data_with_gvi)
    cleaned_data_with_names['cluster'] = cluster_labels
    # Write to new file
    cleaned_data_with_names.to_csv('game_data_with_clusters.csv', index=False)

# Master Function
def run_elbow_method(game_data_with_gvi, max_clusters):
    cleaned_data = clean_data(game_data_with_gvi)
    distortions = elbow_method(cleaned_data, max_clusters)
    plot_elbow(distortions, max_clusters)

# run_elbow_method('game_data_with_gvi.csv', 10)

# Perform training and inference with four clusters
# game_data = append_cluster_labels('game_data_with_gvi.csv', 4)

# Visualize the clusters from game_data_with_clusters
# game_data = pd.read_csv('game_data_with_clusters.csv')
# plt.scatter(game_data['critic_rating'], game_data['GVI'], c=game_data['cluster'], cmap='viridis')
# plt.xlabel('Duration')
# plt.ylabel('GVI')
# plt.title('Clusters of Games')

# plt.show()