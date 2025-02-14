import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
import argparse  # Import argparse
import logging  # Import logging

import pandas as pd
import toml
from sbrscrape import Scoreboard

# Configure logging
logging.basicConfig(
    filename='get_odds_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the parser
parser = argparse.ArgumentParser(description='Fetch NBA odds data for specified seasons.')

# Add the --seasons argument
parser.add_argument(
    '--seasons',
    nargs='+',  # Allows multiple season inputs
    help='List of seasons to process (e.g., 2024-25). If not specified, all seasons in config will be processed.'
)

# Parse the arguments
args = parser.parse_args()

# Adjust the system path
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# Define sportsbook
sportsbook = 'fanduel'

# Load configuration
config = toml.load("../../config.toml")

# Connect to the SQLite database
con = sqlite3.connect("../../Data/OddsData.sqlite")

# Determine which seasons to process
if args.seasons:
    # Validate that the specified seasons exist in the config
    seasons_to_process = []
    for season in args.seasons:
        if season in config['get-odds-data']:
            seasons_to_process.append(season)
        else:
            logging.warning(f"Season '{season}' not found in config. Skipping.")
            print(f"Warning: Season '{season}' not found in config. Skipping.")
else:
    # If no seasons are specified, process all seasons in the config
    seasons_to_process = list(config['get-odds-data'].keys())

# Process each specified season
for key in seasons_to_process:
    value = config['get-odds-data'][key]
    date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date_str = value['end_date'].lower()

    # Handle 'today' as end_date if specified
    if end_date_str == "today":
        end_date = datetime.today().date()
    else:
        end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()

    teams_last_played = {}
    season_df_data = []  # Initialize an empty list to accumulate data for the season

    while date_pointer <= end_date:
        logging.info(f"Getting odds data for {key} on {date_pointer}")
        print(f"Getting odds data for {key} on {date_pointer}")
        sb = Scoreboard(date=date_pointer)

        if not hasattr(sb, "games"):
            logging.info(f"No games found for {date_pointer}. Skipping.")
            print(f"No games found for {date_pointer}. Skipping.")
            date_pointer += timedelta(days=1)
            continue

        for game in sb.games:
            # Process home team
            if game['home_team'] not in teams_last_played:
                teams_last_played[game['home_team']] = date_pointer
                home_games_rested = timedelta(days=7)  # Start of season
            else:
                current_date = date_pointer
                home_games_rested = current_date - teams_last_played[game['home_team']]
                teams_last_played[game['home_team']] = current_date

            # Process away team
            if game['away_team'] not in teams_last_played:
                teams_last_played[game['away_team']] = date_pointer
                away_games_rested = timedelta(days=7)  # Start of season
            else:
                current_date = date_pointer
                away_games_rested = current_date - teams_last_played[game['away_team']]
                teams_last_played[game['away_team']] = current_date

            try:
                season_df_data.append({
                    'Date': date_pointer.strftime("%Y-%m-%d"),
                    'Home': game['home_team'],
                    'Away': game['away_team'],
                    'OU': game['total'][sportsbook],
                    'Spread': game['away_spread'][sportsbook],
                    'ML_Home': game['home_ml'][sportsbook],
                    'ML_Away': game['away_ml'][sportsbook],
                    'Points': game['away_score'] + game['home_score'],
                    'Win_Margin': game['home_score'] - game['away_score'],
                    'Days_Rest_Home': home_games_rested.days,
                    'Days_Rest_Away': away_games_rested.days
                })
            except KeyError:
                logging.warning(f"No {sportsbook} odds data found for game: {game}")
                print(f"No {sportsbook} odds data found for game: {game}")

        # Increment the date
        date_pointer += timedelta(days=1)

        # Sleep for a random interval to avoid overwhelming the server
        time.sleep(random.randint(1, 3))

    # After processing all dates for the season, save the accumulated data
    if season_df_data:
        df = pd.DataFrame(season_df_data)
        table_name = f"odds_{key}_new"
        try:
            df.to_sql(table_name, con, if_exists="replace")
            logging.info(f"Data for {table_name} inserted successfully.")
            print(f"Data for {table_name} inserted successfully.")
        except Exception as e:
            logging.error(f"Error inserting data for {table_name}: {e}")
            print(f"Error inserting data for {table_name}: {e}")
    else:
        logging.info(f"No data collected for season {key}.")
        print(f"No data collected for season {key}.")

# Close the database connection
con.close()
logging.info("Database connection closed.")
print("Database connection closed.")