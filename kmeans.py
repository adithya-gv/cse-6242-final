from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from kneed import KneeLocator

CLEANED_DATA_FILEPATH = 'data/cleaned_data.csv'
CLEANED_DATA_NAMED_FILEPATH = 'data/cleaned_data_with_names.csv'

ALL_FEATURES = ['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']

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
        pd.DataFrame(scaled_data, columns=ALL_FEATURES).to_csv(CLEANED_DATA_FILEPATH, index=False)
    return scaled_data

def clean_data_with_names(game_data_with_gvi, write=False):
    game_data = pd.read_csv(game_data_with_gvi)
    cleaned_data = game_data.dropna(subset=ALL_FEATURES)
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], 0.0)
    cleaned_data = cleaned_data[['game_name'] + ALL_FEATURES]
    if write:
        cleaned_data.to_csv(CLEANED_DATA_NAMED_FILEPATH, index=False)

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
    if method == "knee":
        knee = KneeLocator(range(2, max_clusters + 1), distortions, curve="convex", direction="decreasing").knee
        print(f"Optimal k (Elbow Method): {knee}")
        return knee if knee is not None else 3  
    elif method == "silhouette":
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2 
        print(f"Optimal k (Silhouette Score): {optimal_k_silhouette}")
        return optimal_k_silhouette
    else:
        raise ValueError("Invalid method. Use 'knee' or 'silhouette'.")

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
    distortions, silhouette_scores = elbow_method(cleaned_data, max_clusters)
    # plot_elbow(distortions, silhouette_scores, max_clusters)
    optimal_k = get_optimal_clusters(distortions, silhouette_scores, max_clusters, method)
    
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
    cleaned_data = pd.read_csv(CLEANED_DATA_FILEPATH)
    cleaned_data_with_names = pd.read_csv(CLEANED_DATA_NAMED_FILEPATH)

    distortions, silhouette_scores = elbow_method(cleaned_data, max_clusters=10)
    optimal_k = get_optimal_clusters(distortions, silhouette_scores, max_clusters=10, method="knee")
    kmeans = train_kmeans(cleaned_data, optimal_k)
    cluster_labels = get_cluster_labels(cleaned_data, kmeans)
    cleaned_data_with_names['cluster'] = cluster_labels
    return cleaned_data_with_names

# Example usage
if __name__ == "__main__":
    features = ALL_FEATURES
    game_data_with_gvi = 'data/game_data_with_gvi.csv'
    
    # Run clustering and determine optimal clusters
    generate_cluster_dataset(game_data_with_gvi, features, max_clusters=10, method="knee")
