import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px

# Load data
df = pd.read_csv('game_data_with_clusters_2.csv')

# Define features for filtering
features = ['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

# Normalize features for PCA
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# Initialize Dash app with external Bootstrap stylesheet for better styling
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Game Similarity Visualization"

# Sort the game names alphabetically
sorted_games = sorted(df['game_name'].dropna().unique())

# Set a default selected game (optional)
default_game = sorted_games[0] if sorted_games else None

# Define consistent category order for Plotly
category_order = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Selected Game']  # Updated category order

# Define color mapping for clusters and selected game
color_discrete_map = {
    'Cluster 0': '#1f77b4',  # Blue
    'Cluster 1': '#2ca02c',  # Green
    'Cluster 2': '#ff7f0e',  # Orange
    'Cluster 3': '#9467bd',  # Purple
    'Cluster 4': '#e377c2',  # Pink
    'Cluster 5': '#7f7f7f',  # Grey
    'Selected Game': '#d62728',  # Red
}

# Alternative color mapping when cluster colors are turned off
alternative_color_discrete_map = {
    'Cluster 0': '#1f77b4',  # Blue
    'Cluster 1': '#1f77b4',  # Blue
    'Cluster 2': '#1f77b4',  # Blue
    'Cluster 3': '#1f77b4',  # Blue
    'Cluster 4': '#1f77b4',  # Blue
    'Cluster 5': '#1f77b4',  # Blue
    'Selected Game': '#d62728',  # Red
}

# Layout
# Layout with better positioning
# Adjusted Layout for Left Controls and Right Visualization
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Game Similarity Visualization",
                    style={"text-align": "center", "margin-bottom": "20px"},
                ),
                width=12,
            )
        ),

        # Main Row: Left (Controls + Metadata) and Right (Visualization)
        dbc.Row(
            [
                # Left Column: Controls and Metadata
                dbc.Col(
                    [
                        # Feature Selection
                        html.Label(
                            "Select Features to Include:",
                            style={"font-weight": "bold", "margin-bottom": "5px"},
                        ),
                        dcc.Checklist(
                            id='feature-checklist',
                            options=[
                                {'label': feature, 'value': feature}
                                for feature in features
                            ],
                            value=features,  # Default: all features selected
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "10px"},
                        ),
                        html.Br(),

                        # Game Selection
                        html.Label(
                            "Choose a Game:",
                            style={"font-weight": "bold", "margin-top": "15px", "margin-bottom": "5px"},
                        ),
                        dcc.Dropdown(
                            id='game-dropdown',
                            options=[
                                {'label': game, 'value': game}
                                for game in sorted_games
                            ],
                            value=default_game,  # Set the first game as default
                            placeholder="Select a game",
                            style={"width": "100%"},
                        ),
                        html.Br(),

                        # Cluster Color Toggle
                        dbc.Checkbox(
                            id='toggle-cluster-colors',
                            label="Turn cluster color off",
                            value=False,
                            style={"margin-bottom": "15px", "margin-top": "10px"},
                        ),

                        # Game Filter Options
                        html.Label(
                            "Filter Games:",
                            style={"font-weight": "bold", "margin-bottom": "5px"},
                        ),
                        dcc.RadioItems(
                            id='filter-options',
                            options=[
                                {'label': 'Selected game only', 'value': 'show_selected'},
                                {'label': 'Selected games and same cluster games', 'value': 'show_cluster'},
                                {'label': 'All games', 'value': 'show_all'},
                            ],
                            value='show_all',  # Default: Show all games
                            labelStyle={"display": "block", "margin-bottom": "5px"},
                        ),
                        html.Br(),

                        # Metadata
                        html.H4(
                            "Selected Game Metadata",
                            style={"text-align": "center", "margin-top": "20px"},
                        ),
                        html.Div(
                            id='metadata-container',
                            style={
                                "border": "1px solid #ddd",
                                "padding": "10px",
                                "border-radius": "5px",
                                "height": "30vh",
                                "overflowY": "auto",
                                "background-color": "#f9f9f9",
                                "font-size": "14px",
                            },
                        ),
                    ],
                    md=3,  # Left column width
                    style={"padding-right": "15px"},  # Spacing to separate from plot
                ),

                # Right Column: Visualization
                dbc.Col(
                    html.Div(
                        id='visualization-container',
                        style={"height": "80vh"},  # Full height for the visualization
                    ),
                    md=9,  # Right column width
                ),
            ]
        ),
    ],
    fluid=True,
)



# Callback for dynamic visualization
@app.callback(
    Output('visualization-container', 'children'),
    [
        Input('feature-checklist', 'value'),
        Input('game-dropdown', 'value'),
        Input('toggle-cluster-colors', 'value'),  # Existing Input for the color toggle
        Input('filter-options', 'value'),         # New Input for the filter options
    ],
)
def update_visualization(selected_features, selected_game, toggle_cluster_colors, filter_option):
    if not selected_features:
        return html.Div(
            "Please select at least one feature.",
            style={"text-align": "center", "color": "red"},
        )

    # Determine which color map to use
    if toggle_cluster_colors:
        current_color_map = alternative_color_discrete_map
    else:
        current_color_map = color_discrete_map

    # Start with the full dataset
    filtered_df = df[selected_features + ['game_name', 'cluster']].copy()

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

    # Convert 'highlight' to categorical with ordered categories
    filtered_df['highlight'] = pd.Categorical(
        filtered_df['highlight'],
        categories=category_order,  # Enforce category order
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
            color_discrete_map=current_color_map,  # Use the selected color map
            hover_data=['game_name'],
            category_orders={'highlight': category_order},  # Enforce category order
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
            color_discrete_map=current_color_map,  # Use the selected color map
            labels={
                selected_features[0]: selected_features[0],
                selected_features[1]: selected_features[1],
            },
            hover_data=['game_name'],
            category_orders={'highlight': category_order},  # Enforce category order
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
            color_discrete_map=current_color_map,  # Use the selected color map
            labels={
                selected_features[0]: selected_features[0],
                selected_features[1]: selected_features[1],
                selected_features[2]: selected_features[2],
            },
            hover_data=['game_name'],
            category_orders={'highlight': category_order},  # Enforce category order
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
        pca_df['game_name'] = df['game_name']
        pca_df['cluster'] = df['cluster']

        # Apply filtering based on the selected option
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

        # Convert 'highlight' to categorical with ordered categories
        pca_df['highlight'] = pd.Categorical(
            pca_df['highlight'],
            categories=category_order,  # Enforce category order
            ordered=True
        )

        fig = px.scatter_3d(
            pca_df,
            x='PCA 1',
            y='PCA 2',
            z='PCA 3',
            color='highlight',
            title="3D Scatter Plot (PCA - First 3 Components)",
            color_discrete_map=current_color_map,  # Use the selected color map
            hover_data=['game_name'],
            category_orders={'highlight': category_order},  # Enforce category order
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
                xaxis_title='PCA 1',
                yaxis_title='PCA 2',
                zaxis_title='PCA 3',
            ),
            template='plotly_white',
        )

    return dcc.Graph(figure=fig, style={"height": "100%", "width": "100%"})


# Callback to update metadata display
@app.callback(
    Output('metadata-container', 'children'),
    [
        Input('game-dropdown', 'value'),
    ],
)
def update_metadata(selected_game):
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


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
