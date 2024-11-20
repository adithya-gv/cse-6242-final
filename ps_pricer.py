from psnawp_api import PSNAWP
from psnawp_api.models import SearchDomain
import requests
from bs4 import BeautifulSoup

token = 'lt7u03XwmjSl3ttg5q7rpIG4poWZhkVtWcwQDyq7UAop3hibQCJjqCWek8wUoAl2'

psnawp = PSNAWP(token)

def find_game(game_query):
    search = psnawp.search(search_query=game_query, search_domain=SearchDomain.FULL_GAMES)
    for search_result in search:
        if game_query.lower() == search_result["result"]["invariantName"].lower():
            game_id = search_result["result"]["defaultProduct"]["id"]
            return game_id


def search_game_in_store(game_name):
    """
    Search the PlayStation Store for the game name and retrieve its URL and price.
    """
    id = find_game(game_name)
    search_url = f"https://store.playstation.com/en-us/product/{id}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the price on the product page
        price_element = soup.find('span', class_="psw-t-title-m")  # This class may vary

        if price_element:
            price = price_element.text.strip()
            
            # If its free, convert to 0
            if price.lower() == "free":
                return 0
            else:
                # Convert to float
                return float(price.replace("$", ""))


    except requests.exceptions.RequestException as e:
        return -1