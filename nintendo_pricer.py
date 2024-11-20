from nintendeals import noa
import re

def get_uuid(game_str):
    for game in noa.list_switch_games():
        cleaned_title = re.sub(r'[^\x00-\x7F]+', '', game.title.lower())
        if game_str.lower() == cleaned_title:
            return game.nsuid

def get_ns_game_price(game_str):
    nsuid = str(get_uuid(game_str))
    price = noa.game_info(nsuid).price(country="US")

    # get rid of the leading "USD " and convert to float
    return price.value