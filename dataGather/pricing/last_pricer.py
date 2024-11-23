import requests

def get_game_price(game_name):

    url = 'https://www.cheapshark.com/api/1.0/games'
    params = {'title': game_name}

    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        datapoint = data[0]
        return datapoint['cheapest']
    else:
        return -1