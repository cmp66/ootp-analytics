park_adjustments = {
    "TGS": {
        "Anaheim Angels": -0.94,
        "Arizona Diamondbacks": -0.8,
        "Atlanta Hammers": 1.92,
        "Baltimore Orioles": -2.38,
        "Boston Red Sox": 0.67,
        "Chicago Cubs": -0.82,
        "Chicago White Sox": 0.03,
        "Cincinnati Reds": -0.82,
        "Cleveland Guardians": 3.72,
        "Colorado Rockies": -6.06,
        "Detroit Tigers": 0.5,
        "Florida Marlins": 2.2,
        "Houston Astros": -1.66,
        "Kansas City Monarchs": -0.25,
        "Kansas City Royals": 0.52,
        "Los Angeles Dodgers": 3.45,
        "Milwaukee Brewers": 0.61,
        "Minnesota Twins": -0.25,
        "Montreal Expos": 1.09,
        "New York Mets": -6.06,
        "New York Yankees": -0.13,
        "Oakland Athletics": -2.38,
        "Philadelphia Phillies": 3.45,
        "Pittsburgh Pirates": 1.18,
        "San Diego Padres": 2.97,
        "San Francisco Giants": 3.45,
        "Seattle Mariners": -0.82,
        "St. Louis Cardinals": -6.06,
        "Tampa Bay Devil Rays": 0.51,
        "Texas Rangers": 3.72,
        "Toronto Blue Jays": -0.11,
        "Washington Nationals": -0.8,
    }
}


def get_park_adjustments(league: str) -> dict[str, float]:
    """
    Get the park adjustments for the league

    Args:
    league: str: The league to get the park adjustments for

    Returns:
    dict[str, float]: The park adjustments for the league
    """

    return park_adjustments[league]
