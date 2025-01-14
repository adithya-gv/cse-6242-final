from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from kneed import KneeLocator

SCALED_DATA_FILEPATH = 'data/scaled_data.csv'
RAW_DATA_FILEPATH = 'data/raw_data.csv'

ALL_FEATURES = ['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI', 'genre']

def train_kmeans(cleaned_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(cleaned_data)
    return kmeans

def inference_kmeans(cleaned_data, kmeans):
    return kmeans.predict(cleaned_data)

def clean_data(game_data_with_gvi, write=False):
    game_data = pd.read_csv(game_data_with_gvi)
    cleaned_data = game_data.dropna(subset=ALL_FEATURES)
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0.0)
    cleaned_data = cleaned_data[ALL_FEATURES]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    if write:
        pd.DataFrame(scaled_data, columns=ALL_FEATURES).to_csv(SCALED_DATA_FILEPATH, index=False)
    return scaled_data

def clean_data_with_names(game_data_with_gvi, write=False):
    game_data = pd.read_csv(game_data_with_gvi)
    cleaned_data = game_data.dropna(subset=ALL_FEATURES)
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0.0)
    cleaned_data = cleaned_data[['game_name'] + ALL_FEATURES + ['price']]
    if write:
        cleaned_data.to_csv(RAW_DATA_FILEPATH, index=False)

    return cleaned_data

def elbow_method(cleaned_data, max_clusters, method="knee"):
    distortions = []
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = train_kmeans(cleaned_data, i)
        distortions.append(kmeans.inertia_)
        if method != "knee":
            silhouette_scores.append(silhouette_score(cleaned_data, kmeans.labels_))
    return distortions, silhouette_scores

def get_optimal_clusters(distortions, silhouette_scores, max_clusters, method="knee"):
    knee_val = 3
    optimal_k_silhouette = None
    knee = KneeLocator(range(2, max_clusters + 1), distortions, curve="convex", direction="decreasing").knee
    print(f"Optimal k (Elbow Method): {knee}")
    knee_val = knee 
    if method != "knee":
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2 
        print(f"Optimal k (Silhouette Score): {optimal_k_silhouette}")

    return knee_val


def get_cluster_labels(cleaned_data, kmeans):
    return inference_kmeans(cleaned_data, kmeans)

def plot_elbow(distortions, silhouette_scores, max_clusters, method="knee"):
    plt.figure(figsize=(12, 5))
    
    # Plot WCSS (Elbow)
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (WCSS)')
    
    if method != "knee":
        # Plot Silhouette Scores
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Scores for Different k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.show()

def generate_cluster_dataset(game_data_with_gvi, features, max_clusters=10, method="knee"):
    cleaned_data = clean_data(game_data_with_gvi, write=True)
    
    # Determine optimal number of clusters
    distortions, silhouette_scores = elbow_method(cleaned_data, max_clusters, method=method)
    optimal_k = get_optimal_clusters(distortions, silhouette_scores, max_clusters, method)
    plot_elbow(distortions, silhouette_scores, max_clusters, method=method)
    
    # Train KMeans with the optimal number of clusters
    kmeans = train_kmeans(cleaned_data, optimal_k)

    # Save Model
    with open('kmeans_model.pkl', 'wb') as file:
        pickle.dump(kmeans, file)
    
    # Get Cluster Labels
    cluster_labels = get_cluster_labels(cleaned_data, kmeans)
    cleaned_data_with_names = clean_data_with_names(game_data_with_gvi, write=True)
    cleaned_data_with_names['cluster'] = cluster_labels
    
    # Save to CSV
    cleaned_data_with_names.to_csv('final_data_with_clusters.csv', index=False)
    print(f"Clustered data saved to 'final_data_with_clusters.csv'")


def live_clustering(features):
    cleaned_data = pd.read_csv(SCALED_DATA_FILEPATH)[features]
    cleaned_data_with_names = pd.read_csv(RAW_DATA_FILEPATH)[['game_name'] + features]

    distortions, silhouette_scores = elbow_method(cleaned_data, max_clusters=10)
    optimal_k = get_optimal_clusters(distortions, silhouette_scores, max_clusters=10, method="knee")
    kmeans = train_kmeans(cleaned_data, optimal_k)
    cluster_labels = get_cluster_labels(cleaned_data, kmeans)
    cleaned_data_with_names['cluster'] = cluster_labels
    return cleaned_data_with_names

# Example usage
if __name__ == "__main__":
    features = ALL_FEATURES
    game_data_with_gvi = 'data/final_game_data.csv'
    
    # Run clustering and determine optimal clusters
    generate_cluster_dataset(game_data_with_gvi, features, max_clusters=10, method="both")
