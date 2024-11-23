import rawg_stats as rawg_stats
import pandas as pd
from pricing.agg_pricer import game_price

def computeGVI(name: str):
    print(name)
    try:
        stats = rawg_stats.get_game_stats(name)

        playtime = stats['duration']
        critic_rating = stats['critic_rating']
        peer_rating = stats['peer_rating']
        popularity = stats['popularity']
        year = stats['year']
        platform = stats['platform']

        df = pd.read_csv("data/yearly_game_summary.csv")

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
        price = game_price(name, platform)

        # Compute the Game Value Index
        gvi = (playtime_z + critic_z + peer_z + popularity_z) 
        return gvi, price
    except Exception as e:
        print(e)
        return None, 60
    

# Compute GVI and price for all games, and save to a CSV file
def compute_all():
    df = pd.read_csv("data/final_game_data.csv")
    df = df.dropna(subset=['game_name'])
    df = df.replace([float('inf'), -float('inf')], 0.0)
    df['GVI'] = 0.0
    df['price'] = 0.0

    for i, row in df.iterrows():
        name = row['game_name']
        gvi, price = computeGVI(name)
        print(gvi)
        df.at[i, 'GVI'] = gvi
        df.at[i, 'price'] = price

    df.to_csv("real_final_game_data.csv", index=False)

compute_all()