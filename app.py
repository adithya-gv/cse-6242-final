# app.py
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px
from dash.exceptions import PreventUpdate

from kmeans import ALL_FEATURES
from utils import (
    recluster_data,
    create_figure,
    update_metadata,
    update_game_dropdown_on_click
)  # Import functions from utils.py

CLEANED_DATA_FILEPATH = 'data/cleaned_data.csv'
CLEANED_DATA_NAMED_FILEPATH = 'data/cleaned_data_with_names.csv'

# Initialize Dash app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Game Similarity Visualization"

# Load and preprocess data
game_df = pd.read_csv(CLEANED_DATA_NAMED_FILEPATH)

# Get sorted game names and default game
sorted_games = sorted(game_df['game_name'].dropna().unique())
default_game = sorted_games[0] if sorted_games else None


# Layout
app.layout = dbc.Container(
    [
        dcc.Store(id='intermediate-data', storage_type='memory'),

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
                                for feature in ALL_FEATURES
                            ],
                            value=ALL_FEATURES,  # Default: all features selected
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
                    dcc.Graph(
                        id='main-graph',  # Assign a unique ID here
                        style={"height": "80vh"},  # Full height for the visualization
                    ),
                    md=9,  # Right column width
                ),
            ]
        ),
    ],
    fluid=True,
)

# Callback to update intermediate data
@app.callback(
    Output('intermediate-data', 'data'),
    [Input('feature-checklist', 'value')]
)
def update_intermediate_data(selected_features):
    df, df_scaled = recluster_data(selected_features)
    return {'df': df.to_json(orient='split'), 'df_scaled': df_scaled.to_json(orient='split')}

# Callback to update visualization
@app.callback(
    Output('main-graph', 'figure'),
    [
        Input('intermediate-data', 'data'),
        Input('feature-checklist', 'value'),
        Input('game-dropdown', 'value'),
        Input('toggle-cluster-colors', 'value'),
        Input('filter-options', 'value'),
    ],
)
def update_visualization(intermediate_data, selected_features, selected_game, toggle_cluster_colors, filter_option):
    df = pd.read_json(intermediate_data['df'], orient='split')
    df_scaled = pd.read_json(intermediate_data['df_scaled'], orient='split')
    figure = create_figure(df, selected_features, selected_game, toggle_cluster_colors, filter_option, df_scaled)
    return figure

# Callback for metadata
@app.callback(
    Output('metadata-container', 'children'),
    [Input('game-dropdown', 'value'), Input('intermediate-data', 'data')]
)
def update_metadata_callback(selected_game, intermediate_data):
    df = pd.read_json(intermediate_data['df'], orient='split')
    return update_metadata(df, selected_game)


# Callback for graph click (unchanged)
@app.callback(
    Output('game-dropdown', 'value'),
    [Input('main-graph', 'clickData')],
    [State('game-dropdown', 'value')]
)
def update_dropdown_on_click(clickData, current_game):
    return update_game_dropdown_on_click(clickData, current_game)

if __name__ == '__main__':
    app.run_server(debug=True)