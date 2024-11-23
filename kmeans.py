from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

OPTIMAL_CLUSTER = 6

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
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0.0)
    cleaned_data = cleaned_data[['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data

def clean_data_with_names(game_data_with_gvi):
    game_data = pd.read_csv(game_data_with_gvi)
    cleaned_data = game_data.dropna(subset=['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI'])
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0.0)
    return cleaned_data[['game_name', 'duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']]

def elbow_method(cleaned_data, max_clusters):
    distortions = []
    for i in range(2, max_clusters):
        print(i)
        kmeans = train_kmeans(cleaned_data, i)
        distortions.append(kmeans.inertia_)
    return distortions

def get_cluster_labels(cleaned_data, kmeans):
    kmeans = inference_kmeans(cleaned_data, kmeans)
    return kmeans.labels_

def plot_elbow(distortions, max_clusters):
    plt.plot(range(2, max_clusters), distortions)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

def generate_cluster_dataset(game_data_with_gvi):
    cleaned_data = clean_data(game_data_with_gvi)
    kmeans = train_kmeans(cleaned_data, OPTIMAL_CLUSTER)
    # Save Model
    with open('kmeans_model.pkl', 'wb') as file:
        pickle.dump(kmeans, file)
    cluster_labels = get_cluster_labels(cleaned_data, kmeans)
    cleaned_data_with_names = clean_data_with_names(game_data_with_gvi)
    cleaned_data_with_names['cluster'] = cluster_labels
    cleaned_data_with_names.to_csv('final_data_with_clusters.csv', index=False)

def run_elbow_method(game_data_with_gvi, max_clusters):
    cleaned_data = clean_data(game_data_with_gvi)
    distortions = elbow_method(cleaned_data, max_clusters)
    plot_elbow(distortions, max_clusters)

def live_clustering():
    game_data_with_gvi = 'final_game_data.csv'
    cleaned_data = clean_data(game_data_with_gvi)
    with open('kmeans_model.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    cluster_labels = get_cluster_labels(cleaned_data, kmeans)
    cleaned_data_with_names = clean_data_with_names(game_data_with_gvi)
    cleaned_data_with_names['cluster'] = cluster_labels
    return cleaned_data_with_names

generate_cluster_dataset('data/final_game_data.csv')