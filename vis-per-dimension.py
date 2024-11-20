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
df = pd.read_csv('game_data_with_gvi_mini.csv')

# Define features for filtering
features = ['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

# Normalize features for PCA
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# Initialize Dash app with external stylesheet for better styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = [dbc.themes.JOURNAL]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Game Similarity Visualization"

# Sort the game names alphabetically
sorted_games = sorted(df['game_name'].dropna().unique())

# Set a default selected game (optional)
default_game = sorted_games[0] if sorted_games else None

# Define consistent category order for Plotly
category_order = ['Other Games', 'Selected Game']  # "Selected Game" is last

# Layout
app.layout = html.Div(
    [
        html.H1("Game Similarity Visualization", style={"text-align": "center"}),

        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Select Features to Include:",
                            style={"font-weight": "bold"},
                        ),
                        dcc.Checklist(
                            id='feature-checklist',
                            options=[
                                {'label': feature, 'value': feature}
                                for feature in features
                            ],
                            value=features,  # Default: all features selected
                            inline=True,
                            inputStyle={
                                "margin-right": "5px",
                                "margin-left": "10px",
                            },
                        ),
                    ],
                    className='six columns',
                    style={"margin-bottom": "20px"},
                ),

                html.Div(
                    [
                        html.Label(
                            "Choose a Game:",
                            style={"font-weight": "bold"},
                        ),
                        dcc.Dropdown(
                            id='game-dropdown',
                            options=[
                                {'label': game, 'value': game}
                                for game in sorted_games  # Use the sorted list here
                            ],
                            value=default_game,  # Set the first game as default
                            placeholder="Select a game",
                            style={"width": "100%", "margin-bottom": "20px"},
                        ),
                    ],
                    className='six columns',
                ),
            ],
            className='row',
        ),

        html.Div(
            id='visualization-container',
            style={"height": "72vh"},
        ),
    ],
    style={"padding": "20px"},
)

# Callback for dynamic visualization
@app.callback(
    Output('visualization-container', 'children'),
    [
        Input('feature-checklist', 'value'),
        Input('game-dropdown', 'value'),
    ],
)
def update_visualization(selected_features, selected_game):
    if not selected_features:
        return html.Div(
            "Please select at least one feature.",
            style={"text-align": "center", "color": "red"},
        )

    # Filter dataset based on selected features
    filtered_df = df[selected_features].copy()
    filtered_df['game_name'] = df['game_name']

    # Highlight the selected game
    if selected_game is None:
        # If no game is selected, categorize all as 'Other Games'
        filtered_df['highlight'] = 'Other Games'
    else:
        filtered_df['highlight'] = filtered_df['game_name'].apply(
            lambda x: 'Selected Game' if x == selected_game else 'Other Games'
        )

    # Convert 'highlight' to categorical with ordered categories
    filtered_df['highlight'] = pd.Categorical(
        filtered_df['highlight'],
        categories=category_order,  # Enforce category order
        ordered=True
    )

    # Adjust visualization based on the number of features selected
    if len(selected_features) == 1:
        # Single feature: Box plot to show distribution
        feature = selected_features[0]
        fig = px.box(
            filtered_df,
            y=feature,
            points="all",
            color='highlight',
            title=f"Distribution of {feature}",
            color_discrete_map={
                'Selected Game': 'red',
                'Other Games': 'lightblue',
            },
            hover_data=['game_name'],
            category_orders={'highlight': category_order},  # Enforce category order
        )
        fig.update_layout(
            yaxis_title=feature,
            legend_title_text='Game Selection',
            title_font_size=20,
            legend_title_font_size=14,
            legend_font_size=12,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            template='plotly_white',
        )
        fig.update_traces(jitter=0.3, marker_size=8)

    elif len(selected_features) == 2:
        # Two features: 2D scatter plot
        fig = px.scatter(
            filtered_df,
            x=selected_features[0],
            y=selected_features[1],
            color='highlight',
            title="2D Scatter Plot",
            color_discrete_map={
                'Selected Game': 'red',
                'Other Games': 'lightblue',
            },
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
            legend_title_text='Game Selection',
            title_font_size=20,
            legend_title_font_size=14,
            legend_font_size=12,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            template='plotly_white',
        )

    elif len(selected_features) == 3:
        # Three features: 3D scatter plot
        fig = px.scatter_3d(
            filtered_df,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='highlight',
            title="3D Scatter Plot",
            color_discrete_map={
                'Selected Game': 'red',
                'Other Games': 'lightblue',
            },
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
            legend_title_text='Game Selection',
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

        if selected_game is None:
            pca_df['highlight'] = 'Other Games'
        else:
            pca_df['highlight'] = pca_df['game_name'].apply(
                lambda x: 'Selected Game' if x == selected_game else 'Other Games'
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
            color_discrete_map={
                'Selected Game': 'red',
                'Other Games': 'lightblue',
            },
            hover_data=['game_name'],
            category_orders={'highlight': category_order},  # Enforce category order
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            legend_title_text='Game Selection',
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
