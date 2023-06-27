"""
This script downloads and unzips the data from Google Drive into a specified data directory.
"""

import os

import requests

ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")

HISTORICAL_DATA_FILENAME = "a1_RestaurantReviews_HistoricDump.tsv"
FRESH_DATA_FILENAME = "a2_RestaurantReviews_FreshDump.tsv"

GDRIVE_ID_HISTORICAL_DATA = "1vY_6GRk1dCko5861mXHrMzDEMI8h4MW6"
GDRIVE_ID_FRESH_DATA = "1ces0S1KSqNHuIh5yZ8B4-ksq3iNLAeL5"


def load_data():
    """
    This function downloads and unzips data from Google Drive into a specified data directory.
    """

    historical_file_path = os.path.join(DATA_FOLDER, HISTORICAL_DATA_FILENAME)
    fresh_file_path = os.path.join(DATA_FOLDER, FRESH_DATA_FILENAME)

    # Create the data directory if it does not exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Create the historical data file if it does not exist
    if not os.path.exists(historical_file_path):
        print("Historical data not found. Downloading from Google Drive...")
        with open(historical_file_path, 'wb') as file:
            data = requests.get(__get_drive_url(GDRIVE_ID_HISTORICAL_DATA), timeout=25)
            file.write(data.content)

    # Create the fresh data file if it does not exist
    if not os.path.exists(fresh_file_path):
        print("Fresh data not found. Downloading from Google Drive...")
        with open(fresh_file_path, 'wb') as file:
            data = requests.get(__get_drive_url(GDRIVE_ID_FRESH_DATA), timeout=25)
            file.write(data.content)

    print("[DATA PHASE COMPLETE]\n")


def __get_drive_url(gdrive_id: str) -> str:
    """
    This Python function takes a Google Drive file ID and returns a URL that can be used to download the
    file.

    :param gdrive_id: The `gdrive_id` parameter is a string that represents the unique identifier of a
    file or folder in Google Drive. It is used to construct a URL that can be used to download the file
    or access the folder
    :type gdrive_id: str
    :return: a string which is the URL of a Google Drive file with the given ID. The URL is constructed
    by appending the ID to the base URL "https://drive.google.com/uc?export=download&id=".
    """
    base_url = r"https://drive.google.com/uc?export=download&id="
    return base_url + gdrive_id


if __name__ == "__main__":
    load_data()
