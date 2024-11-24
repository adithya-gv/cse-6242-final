import io
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd

from kmeans import ALL_FEATURES, RAW_DATA_FILEPATH
from utils import (
    recluster_data,
    create_figure,
    update_metadata,
    update_game_dropdown_on_click,
    recommend_games
)

# Initialize Dash app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Game Recommender and Visualization System"

# Load and preprocess data
full_df = pd.read_csv(RAW_DATA_FILEPATH)

# Get sorted game names and default game
sorted_games = sorted(full_df['game_name'].dropna().unique())
default_game = sorted_games[0] if sorted_games else None

# Layout
app.layout = dbc.Container(
    [
        dcc.Store(id='intermediate-data', storage_type='memory'),
        dcc.Store(id='recommended-games-data', storage_type='memory'),

        # Header
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Game Recommender and Visualization System",
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
                            "Find a Game:",
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

                        # Top 5 Favorite Games Input
                        html.Label(
                            "Enter Your Top 5 Favorite Games:",
                            style={"font-weight": "bold", "margin-top": "15px", "margin-bottom": "5px"},
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id={'type': 'favorite-game-dropdown', 'index': i},
                                    options=[
                                        {'label': game, 'value': game}
                                        for game in sorted_games
                                    ],
                                    placeholder=f"Select favorite game {i+1}",
                                    style={"width": "100%", "margin-bottom": "10px"},
                                    searchable=True,
                                )
                                for i in range(5)
                            ]
                        ),
                        html.Br(),

                        # Price Range Input
                        html.Label(
                            "Enter Price Range:",
                            style={"font-weight": "bold", "margin-top": "15px", "margin-bottom": "5px"},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id='min-price',
                                    type='number',
                                    placeholder='Min Price',
                                    style={"width": "48%", "margin-right": "4%"},
                                ),
                                dcc.Input(
                                    id='max-price',
                                    type='number',
                                    placeholder='Max Price',
                                    style={"width": "48%"},
                                ),
                            ]
                        ),
                        html.Br(),

                        # Submit Button for Favorite Games
                        dbc.Button(
                            "Submit Favorite Games",
                            id='submit-favorite-games',
                            color='primary',
                            style={"margin-bottom": "15px"}
                        ),

                        # Recommended Games Display
                        html.Div(
                            id='recommended-games-container',
                            style={
                                "border": "1px solid #ddd",
                                "padding": "10px",
                                "border-radius": "5px",
                                "margin-top": "15px",
                                "background-color": "#f9f9f9",
                                "font-size": "14px",
                            },
                        ),

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
                                {'label': 'Favorite and recommended games only', 'value': 'show_favorites_recommended'},
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
                                "height": "12vh",
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
                        style={"height": "80vh", "overflow": "visible"},  # Full height for the visualization
                    ),
                    md=9,  # Right column width
                ),
            ]
        ),
    ],
    fluid=True,
)

# Callback to update intermediate data (clustered)
@app.callback(
    Output('intermediate-data', 'data'),
    [Input('feature-checklist', 'value')]
)
def update_intermediate_data(selected_features):
    df = recluster_data(full_df, selected_features)
    return {'df': df.to_json(orient='split')}

# Callback to update visualization
@app.callback(
    Output('main-graph', 'figure'),
    [
        Input('intermediate-data', 'data'),
        Input('feature-checklist', 'value'),
        Input('game-dropdown', 'value'),
        Input('toggle-cluster-colors', 'value'),
        Input('filter-options', 'value'),
        Input('recommended-games-data', 'data'),
        Input('min-price', 'value'),
        Input('max-price', 'value'),
    ],
)
def update_visualization(intermediate_data, selected_features, selected_game, toggle_cluster_colors, filter_option, recommended_games_data, min_price, max_price):
    df = pd.read_json(io.StringIO(intermediate_data['df']), orient='split')
    recommended_games = pd.read_json(io.StringIO(recommended_games_data['recommended_games']), orient='split', typ='series') if recommended_games_data else None
    favorite_games = pd.read_json(io.StringIO(recommended_games_data['favorite_games']), orient='split', typ='series') if recommended_games_data else None
    # Identify recommended and favorite games
    recommended_and_favorite_games = pd.concat([recommended_games, favorite_games]).unique() if recommended_games_data else []

    # Apply price filtering for visualization purposes, excluding recommended and favorite games
    if min_price is not None:
        df = df[(df['price'] >= min_price) | (df['game_name'].isin(recommended_and_favorite_games))]
    if max_price is not None:
        df = df[(df['price'] <= max_price) | (df['game_name'].isin(recommended_and_favorite_games))]

    print("df after price filtering", df)
    figure = create_figure(df, selected_features, selected_game, toggle_cluster_colors, filter_option, recommended_games, favorite_games)
    return figure

# Callback for metadata
@app.callback(
    Output('metadata-container', 'children'),
    [Input('game-dropdown', 'value'), Input('intermediate-data', 'data')]
)
def update_metadata_callback(selected_game, intermediate_data):
    df = pd.read_json(io.StringIO(intermediate_data['df']), orient='split')
    return update_metadata(df, selected_game)

# Callback for graph click
@app.callback(
    Output('game-dropdown', 'value'),
    [Input('main-graph', 'clickData')],
    [State('game-dropdown', 'value')]
)
def update_dropdown_on_click(clickData, current_game):
    return update_game_dropdown_on_click(clickData, current_game)

# Callback to process favorite games input
@app.callback(
    [
        Output('recommended-games-data', 'data'),
        Output('recommended-games-container', 'children')
    ],
    [
        Input('submit-favorite-games', 'n_clicks'),
        Input('intermediate-data', 'data'),
        Input('feature-checklist', 'value'),
        Input('min-price', 'value'),
        Input('max-price', 'value'),
    ],
    [State({'type': 'favorite-game-dropdown', 'index': dash.dependencies.ALL}, 'value')],
    prevent_initial_call=True
)
def process_favorite_games_callback(n_clicks, intermediate_data, selected_features, min_price, max_price, favorite_games):
    if n_clicks:
        # Pretend there's a function in utils to process favorite games
        df = pd.read_json(io.StringIO(intermediate_data['df']), orient='split')
        min_price = 0 if min_price is None else min_price
        max_price = 1000 if max_price is None else max_price
        recommended_games = recommend_games(favorite_games, df, selected_features, min_price, max_price)
        recommended_games_list = recommended_games.to_list()

        # Store the processed data
        recommend_games_data = {'recommended_games': recommended_games.to_json(orient='split'), 'favorite_games': pd.Series(favorite_games).to_json(orient='split')}

        # Create the text output for recommended games
        recommended_games_container = html.Div([
            html.H4("Recommended Games:", style={"margin-top": "10px"}),
            html.Ul([html.Li(game) for game in recommended_games_list])
        ])

        return recommend_games_data, recommended_games_container

    else:
        return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
