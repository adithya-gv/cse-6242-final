import requests

def get_app_id(game_name):
    """
    Given the name of a game, return its Steam app_id.
    """
    # Endpoint to retrieve the list of all games and their app_ids
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    
    try:
        # Request to get the list of all apps
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the JSON response
        apps = response.json()['applist']['apps']
        
        # Search for the game by name, case-insensitive
        for app in apps:
            if app['name'].lower() == game_name.lower():
                return app['appid']
        
        # If not found, return None
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def get_steam_game_price(game_name):
    app_id = get_app_id(game_name)
    # Endpoint to retrieve the price of a game
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    
    try:
        # Request to get the price of the game
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Check if the game is available and has a price
        if data[str(app_id)]['success'] and 'price_overview' in data[str(app_id)]['data']:
            price = data[str(app_id)]['data']['price_overview']['final']
            return price / 100
        
        # If not available or no price, return None
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None