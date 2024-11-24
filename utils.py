import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px
from dash import html
from dash.exceptions import PreventUpdate

from kmeans import live_clustering, ALL_FEATURES

def recluster_data(full_df, features=ALL_FEATURES):
    df = live_clustering(features)
    full_df['cluster'] = df['cluster']
    imputer = SimpleImputer(strategy='mean')
    full_df[features] = imputer.fit_transform(full_df[features])

    scaler = StandardScaler()
    df_scaled = full_df.copy()
    df_scaled[features] = scaler.fit_transform(full_df[features])
    return full_df, df_scaled

def create_figure(df, selected_features, selected_game, toggle_cluster_colors, filter_option, df_scaled, recommended_df):
    cluster_counts = df['cluster'].nunique()
    category_order = [f'Cluster {i}' for i in range(cluster_counts)] + ['Selected Game', 'Recommended Game']

    default_colors = px.colors.qualitative.Plotly
    color_discrete_map = {cat: default_colors[i % len(default_colors)] for i, cat in enumerate(category_order[:-2])}
    color_discrete_map['Selected Game'] = '#d62728'
    color_discrete_map['Recommended Game'] = '#2ca02c'

    alternative_color_discrete_map = {cat: '#1f77b4' for cat in category_order[:-2]}
    alternative_color_discrete_map['Selected Game'] = '#d62728'

    if toggle_cluster_colors:
        current_color_map = alternative_color_discrete_map
    else:
        current_color_map = color_discrete_map

    # Filter and highlight
    filtered_df = filter_and_highlight_data(df, selected_features, selected_game, filter_option, category_order) # Helper function

    # Apply filtering based on the selected option
    if filter_option == 'show_selected':
        if selected_game is not None:
            # Filter to show only the selected game
            filtered_df = filtered_df[filtered_df['game_name'] == selected_game]
    elif filter_option == 'show_cluster':
        if selected_game is not None:
            # Get the cluster of the selected game
            selected_cluster = df.loc[df['game_name'] == selected_game, 'cluster'].values
            if len(selected_cluster) > 0:
                selected_cluster = selected_cluster[0]
                # Filter to show games in the same cluster
                filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]

    # Highlighting logic
    if filter_option == 'show_all':
        if selected_game is None:
            # If no game is selected, categorize all based on their cluster
            filtered_df['highlight'] = filtered_df['cluster'].apply(lambda x: f'Cluster {x}')
        else:
            # Highlight the selected game and categorize others by cluster
            filtered_df['highlight'] = filtered_df.apply(
                lambda row: 'Selected Game' if row['game_name'] == selected_game else f'Cluster {row["cluster"]}',
                axis=1
            )
    elif filter_option == 'show_selected':
        if selected_game is not None:
            # All entries are the selected game
            filtered_df['highlight'] = 'Selected Game'
        else:
            # No game selected, this case shouldn't typically occur
            filtered_df['highlight'] = 'Selected Game'
    elif filter_option == 'show_cluster':
        if selected_game is not None:
            # Highlight the selected game, others are in the same cluster
            filtered_df['highlight'] = filtered_df.apply(
                lambda row: 'Selected Game' if row['game_name'] == selected_game else f'Cluster {row["cluster"]}',
                axis=1
            )
        else:
            # No game selected, categorize all based on their cluster
            filtered_df['highlight'] = filtered_df['cluster'].apply(lambda x: f'Cluster {x}')
    else:
        # Default behavior
        if selected_game is None:
            filtered_df['highlight'] = filtered_df['cluster'].apply(lambda x: f'Cluster {x}')
        else:
            filtered_df['highlight'] = filtered_df.apply(
                lambda row: 'Selected Game' if row['game_name'] == selected_game else f'Cluster {row["cluster"]}',
                axis=1
            )

    # Add recommended games highlighting
    if recommended_df is not None and not recommended_df.empty:
        filtered_df['highlight'] = filtered_df.apply(
            lambda row: 'Recommended Game' if row['game_name'] in recommended_df.values else row['highlight'],
            axis=1
        )

    filtered_df['highlight'] = pd.Categorical(
        filtered_df['highlight'],
        categories=category_order,
        ordered=True
    )

    # Adjust visualization based on the number of features selected
    num_features = len(selected_features)
    if num_features == 1:
        # Single feature: Box plot to show distribution
        feature = selected_features[0]
        fig = px.box(
            filtered_df,
            y=feature,
            points="all",
            color='highlight',
            title=f"Distribution of {feature}",
            color_discrete_map=current_color_map,
            hover_data=filtered_df.columns,
            category_orders={'highlight': category_order},
        )
        fig.update_layout(
            yaxis_title=feature,
            legend_title_text='Game Category',
            title_font_size=20,
            legend_title_font_size=14,
            legend_font_size=12,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            template='plotly_white',
        )
        fig.update_traces(jitter=0.3, marker_size=8)

    elif num_features == 2:
        # Two features: 2D scatter plot
        fig = px.scatter(
            filtered_df,
            x=selected_features[0],
            y=selected_features[1],
            color='highlight',
            title="2D Scatter Plot",
            color_discrete_map=current_color_map,
            labels={
                selected_features[0]: selected_features[0],
                selected_features[1]: selected_features[1],
            },
            hover_data=filtered_df.columns,
            category_orders={'highlight': category_order},
        )
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        fig.update_layout(
            xaxis=dict(showgrid=True, tickformat=".2f"),
            yaxis=dict(showgrid=True, tickformat=".2f"),
            legend_title_text='Game Category',
            title_font_size=20,
            legend_title_font_size=14,
            legend_font_size=12,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            template='plotly_white',
        )

    elif num_features == 3:
        # Three features: 3D scatter plot
        fig = px.scatter_3d(
            filtered_df,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='highlight',
            title="3D Scatter Plot",
            color_discrete_map=current_color_map,
            labels={
                selected_features[0]: selected_features[0],
                selected_features[1]: selected_features[1],
                selected_features[2]: selected_features[2],
            },
            hover_data=filtered_df.columns,
            category_orders={'highlight': category_order},
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            legend_title_text='Game Category',
            title_font_size=20,
            legend_title_font_size=14,
            legend_font_size=12,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="white",
            scene=dict(
                xaxis_title=selected_features[0],
                yaxis_title=selected_features[1],
                zaxis_title=selected_features[2],
            ),
            template='plotly_white',
        )

    else:
        # More than three features: Apply PCA and show first 3 components in 3D scatter plot
        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(df_scaled[selected_features])
        pca_df = pd.DataFrame(pca_results, columns=['PCA 1', 'PCA 2', 'PCA 3'])
        pca_df = pd.concat([pca_df, df.reset_index(drop=True)], axis=1)
        components = pd.DataFrame(pca.components_, columns=selected_features, index=['PC1', 'PC2', 'PC3'])
        dominant_features = components.apply(lambda x: x.nlargest(2).index.tolist(), axis=1)
        x_axis_label = f"PC1 ({', '.join(dominant_features.loc['PC1'])})"
        y_axis_label = f"PC2 ({', '.join(dominant_features.loc['PC2'])})"
        z_axis_label = f"PC3 ({', '.join(dominant_features.loc['PC3'])})"

        if filter_option == 'show_selected':
            if selected_game is not None:
                pca_df = pca_df[pca_df['game_name'] == selected_game]
        elif filter_option == 'show_cluster':
            if selected_game is not None:
                selected_cluster = df.loc[df['game_name'] == selected_game, 'cluster'].values
                if len(selected_cluster) > 0:
                    selected_cluster = selected_cluster[0]
                    pca_df = pca_df[pca_df['cluster'] == selected_cluster]

        # Highlighting logic
        if filter_option == 'show_all':
            if selected_game is None:
                # If no game is selected, categorize all based on their cluster
                pca_df['highlight'] = pca_df['cluster'].apply(lambda x: f'Cluster {x}')
            else:
                # Highlight the selected game and categorize others by cluster
                pca_df['highlight'] = pca_df.apply(
                    lambda row: 'Selected Game' if row['game_name'] == selected_game else f'Cluster {row["cluster"]}',
                    axis=1
                )
        elif filter_option == 'show_selected':
            if selected_game is not None:
                # All entries are the selected game
                pca_df['highlight'] = 'Selected Game'
            else:
                # No game selected, default to cluster categorization
                pca_df['highlight'] = pca_df['cluster'].apply(lambda x: f'Cluster {x}')
        elif filter_option == 'show_cluster':
            if selected_game is not None:
                # Highlight the selected game, others are in the same cluster
                pca_df['highlight'] = pca_df.apply(
                    lambda row: 'Selected Game' if row['game_name'] == selected_game else f'Cluster {row["cluster"]}',
                    axis=1
                )
            else:
                # No game selected, categorize all based on their cluster
                pca_df['highlight'] = pca_df['cluster'].apply(lambda x: f'Cluster {x}')
        else:
            # Default behavior
            if selected_game is None:
                pca_df['highlight'] = pca_df['cluster'].apply(lambda x: f'Cluster {x}')
            else:
                pca_df['highlight'] = pca_df.apply(
                    lambda row: 'Selected Game' if row['game_name'] == selected_game else f'Cluster {row["cluster"]}',
                    axis=1
                )

        if recommended_df is not None and not recommended_df.empty:
            pca_df['highlight'] = pca_df.apply(
                lambda row: 'Recommended Game' if row['game_name'] in recommended_df.values else row['highlight'],
                axis=1
            )

        pca_df['highlight'] = pd.Categorical(
            pca_df['highlight'],
            categories=category_order,
            ordered=True
        )

        fig = px.scatter_3d(
            pca_df,
            x='PCA 1',
            y='PCA 2',
            z='PCA 3',
            color='highlight',
            title="3D Scatter Plot (PCA - First 3 Components)",
            color_discrete_map=current_color_map,
            hover_data=pca_df.columns,
            category_orders={'highlight': category_order},
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            legend_title_text='Game Category',
            title_font_size=20,
            legend_title_font_size=14,
            legend_font_size=12,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="white",
            scene=dict(
                xaxis_title=x_axis_label,
                yaxis_title=y_axis_label,
                zaxis_title=z_axis_label,
            ),
            template='plotly_white',
        )

    return fig


def filter_and_highlight_data(df, selected_features, selected_game, filter_option, category_order):
    filtered_df = df[selected_features + ['game_name', 'cluster']].copy()


    if filter_option == 'show_selected':
        if selected_game is not None:
            filtered_df = filtered_df[filtered_df['game_name'] == selected_game]
    elif filter_option == 'show_cluster':
        if selected_game is not None:
            selected_cluster = df.loc[df['game_name'] == selected_game, 'cluster'].iloc[0] if df['game_name'].isin([selected_game]).any() else -1
            if selected_cluster != -1:  # Ensure selected_cluster is valid
                filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]

    # Consistent highlighting logic (simplified)
    if selected_game is not None and any(filtered_df["game_name"].isin([selected_game])): #Check if selected_game exists in filtered data
        filtered_df['highlight'] = filtered_df.apply(lambda row: 'Selected Game' if row['game_name'] == selected_game else f'Cluster {row["cluster"]}', axis=1)
    else:
        filtered_df['highlight'] = filtered_df['cluster'].apply(lambda x: f'Cluster {x}')


    filtered_df['highlight'] = pd.Categorical(filtered_df['highlight'], categories=category_order, ordered=True)

    return filtered_df

def update_metadata(df, selected_game):
    if selected_game is None:
        return html.Div(
            "No game selected.",
            style={"text-align": "center", "color": "#888"},
        )
    
    # Find the cluster of the selected game
    game_info = df[df['game_name'] == selected_game]
    if game_info.empty:
        return html.Div(
            "Selected game not found in the dataset.",
            style={"text-align": "center", "color": "red"},
        )
    
    selected_cluster = game_info.iloc[0]['cluster']
    
    # Count the number of games in the same cluster
    cluster_size = df[df['cluster'] == selected_cluster].shape[0]
    
    # Create the metadata display
    metadata = [
        html.P(f"Game Name: {selected_game}", style={"font-size": "16px"}),
        html.P(f"Cluster: Cluster {selected_cluster}", style={"font-size": "16px"}),
        html.P(f"Cluster Size: {cluster_size} games", style={"font-size": "16px"}),
    ]
    
    return html.Div(metadata)

def recommend_games(favorite_games, df, features):
    recommendations = []

    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        favorite_in_cluster = cluster_df[cluster_df['game_name'].isin(favorite_games)]

        k = len(favorite_in_cluster)
        if k > 0:
            # Calculate centroid of favorite games in this cluster
            centroid = favorite_in_cluster[features].mean().values.reshape(1, -1)

            # Find k closest games to the centroid
            non_favorite_games = cluster_df[~cluster_df['game_name'].isin(favorite_games)]
            if len(non_favorite_games) >= k:
                distances = np.linalg.norm(non_favorite_games[features].values - centroid, axis=1)

                # Get the indices of k closest games
                k_closest_indices = np.argsort(distances)[:k]

                # Retrieve the closest games based on the indices
                closest_games = non_favorite_games.iloc[k_closest_indices]['game_name'].values
            else:
                closest_games = non_favorite_games['game_name'].values

            # Add closest games to recommendations
            recommendations.extend(closest_games)

    return pd.Series(recommendations)



def update_game_dropdown_on_click(clickData, current_game):
    if clickData is None:
        raise PreventUpdate  # No click has occurred
    
    # Extract the game name from the clicked point
    try:
        # Assuming 'game_name' is the first item in customdata
        clicked_game = clickData['points'][0]['customdata'][0]
        if clicked_game != current_game:
            return clicked_game
        else:
            raise PreventUpdate  # Clicked game is already selected
    except (IndexError, KeyError, TypeError):
        raise PreventUpdate  # In case of unexpected clickData structure