import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
import argparse  # Import argparse

import toml

# Initialize the parser
parser = argparse.ArgumentParser(description='Fetch NBA team data for specified seasons.')

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

from src.Utils.tools import get_json_data, to_data_frame

# Load configuration
config = toml.load("../../config.toml")

# Get the data URL from the config
url = config['data_url']

# Connect to the SQLite database
con = sqlite3.connect("../../Data/TeamData.sqlite")

# Determine which seasons to process
if args.seasons:
    # Validate that the specified seasons exist in the config
    seasons_to_process = []
    for season in args.seasons:
        if season in config['get-data']:
            seasons_to_process.append(season)
        else:
            print(f"Warning: Season '{season}' not found in config. Skipping.")
else:
    # If no seasons are specified, process all seasons in the config
    seasons_to_process = list(config['get-data'].keys())

# Process each specified season
for key in seasons_to_process:
    value = config['get-data'][key]
    date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()

    while date_pointer <= end_date:
        print(f"Getting data for {key} on {date_pointer}")

        try:
            # Fetch raw data
            raw_data = get_json_data(
                url.format(
                    date_pointer.month,
                    date_pointer.day,
                    value['start_year'],
                    date_pointer.year,
                    key
                )
            )
            # Convert raw data to DataFrame
            df = to_data_frame(raw_data)

            # Assign the correct date
            df['Date'] = date_pointer.strftime("%Y-%m-%d")

            # Save to SQLite database
            table_name = date_pointer.strftime("%Y-%m-%d")
            df.to_sql(table_name, con, if_exists="replace")

            print(f"Data for {table_name} inserted successfully.")

        except Exception as e:
            print(f"Error fetching data for {date_pointer} in season {key}: {e}")

        # Increment the date
        date_pointer += timedelta(days=1)

        # Sleep for a random interval to avoid overwhelming the server
        time.sleep(random.randint(1, 3))

# Close the database connection
con.close()