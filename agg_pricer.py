from nintendo_pricer import get_ns_game_price
from steam_pricer import get_steam_game_price
from last_pricer import get_game_price
from ps_pricer import search_game_in_store

def _get_game_price(game_name, platform):
    try:
        if platform == "Nintendo":
            return get_ns_game_price(game_name)
        elif platform == "Steam":
            return get_steam_game_price(game_name)
        elif platform == "PS":
            return search_game_in_store(game_name)
        else:
            return get_game_price(game_name)
    except:
        return -1

def determine_platform(platform):
    if "Nintendo" in platform:
        return "Nintendo"
    elif "Steam" in platform:
        return "Steam"
    elif "Playstation" in platform:
        return "PS"
    else:
        return platform

def game_price(game_name, platform):
    plat = determine_platform(platform)
    val = _get_game_price(game_name, plat)
    if val is None:
        return 59.99
    elif val == -1:
        return 59.99
    else:
        return float(val)