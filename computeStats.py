import rawg_stats
import pandas as pd
from nintendo_pricer import get_game_price

def computeGVI(name: str):
    stats = rawg_stats.get_game_stats(name)

    playtime = stats['duration']
    critic_rating = stats['critic_rating']
    peer_rating = stats['peer_rating']
    popularity = stats['popularity']
    year = stats['year']

    df = pd.read_csv("mean_game_stats.csv")

    # Extract the statistics for the specified year
    year_stats = df[df["year"] == year]

    # Extract the mean and standard deviation from the year stats 
    playtime_mean = year_stats["playtime_mean"].values[0]
    playtime_std = year_stats["playtime_std_dev"].values[0]
    critic_mean = year_stats["critic_mean"].values[0]
    critic_std = year_stats["critic_std_dev"].values[0]
    peer_mean = year_stats["peer_mean"].values[0]
    peer_std = year_stats["peer_std_dev"].values[0]
    popularity_mean = year_stats["count_mean"].values[0]
    popularity_std = year_stats["count_std_dev"].values[0]

    # Compute Z-score for each
    playtime_z = (playtime - playtime_mean) / playtime_std
    critic_z = (critic_rating - critic_mean) / critic_std
    peer_z = (peer_rating - peer_mean) / peer_std
    popularity_z = (popularity - popularity_mean) / popularity_std
    
    # Get the game's price
    price = get_game_price(name)

    # Compute the Game Value Index
    gvi = (playtime_z + critic_z + peer_z + popularity_z) / price
    return gvi
    

computeGVI("The Legend of Zelda: Breath of the Wild")