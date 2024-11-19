import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import plotly.express as px
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('game_data_with_gvi_mini.csv')

# Define features for clustering
features = ['duration', 'critic_rating', 'peer_rating', 'popularity', 'GVI']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

# Normalize features for clustering
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df_scaled['cluster'] = kmeans.fit_predict(df_scaled[features])

# Create the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Game Similarity Visualization", style={"text-align": "center", "margin-bottom": "20px"}),
    dcc.Dropdown(
        id='game-dropdown',
        options=[{'label': game, 'value': game} for game in df['game_name'].unique()],
        placeholder="Select a game",
        style={"width": "50%", "margin": "auto", "margin-bottom": "20px"}
    ),
    html.Div(id='scatter-container', style={"height": "90vh"})  # Container to fill 90% of the viewport height
], style={
    "height": "100vh",  # Full height for the app layout
    "margin": "3% 5%",  # Global margin: 2% vertical, 5% horizontal
    "display": "flex",
    "flex-direction": "column",
    "box-sizing": "border-box"  # Ensures margins are included in total layout size
})

# Perform PCA on the scaled features to reduce to 2 dimensions
pca = PCA(n_components=2)
df_scaled[['pca_x', 'pca_y']] = pca.fit_transform(df_scaled[features])

@app.callback(
    Output('scatter-container', 'children'),
    [Input('game-dropdown', 'value')]
)
def update_scatter(selected_game):
    if not selected_game:
        return html.Div("Please select a game from the dropdown.", style={"text-align": "center", "color": "gray"})
    
    # Filter top 100 games
    top_100_games = df_scaled.head(100)
    
    # Add cluster labels to the top 100 games
    top_100_games['cluster'] = df_scaled.loc[top_100_games.index, 'cluster']
    
    # Create scatter plot using PCA components
    fig = px.scatter(
        top_100_games,
        x='pca_x',
        y='pca_y',
        color='cluster',
        hover_data=['game_name', 'peer_rating', 'popularity', 'GVI'],
        title="PCA Scatter Plot of Clusters (Top 100 Games)"
    )
    
    # Update layout to make the plot fill the container
    fig.update_layout(
        height=800,  # Adjust height if needed
        margin=dict(l=0, r=0, t=30, b=0),  # Remove extra margins inside the plot
        paper_bgcolor="white",  # Background of the plot
        plot_bgcolor="white",  # Background of the grid area
    )
    
    fig.update_layout(coloraxis_showscale=False) # hide or unhide the color bar on right

    # Return the graph with the mode bar hidden
    return dcc.Graph(
        figure=fig,
        config={
            "displayModeBar": False,
            },  # Disable mode bar
        style={"height": "100%"}  # Fill parent container
    )

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
