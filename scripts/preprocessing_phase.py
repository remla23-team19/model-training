import os
import re
import sys

import nltk

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

# Define folder locations
__ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
__DATA_FOLDER = os.path.join(__ROOT_FOLDER, "data")
__OUTPUT_FOLDER = os.path.join(__ROOT_FOLDER, "output")

# Define file locations
__FILEPATH_HISTORICAL_DATA = os.path.join(
    __DATA_FOLDER, "a1_RestaurantReviews_HistoricDump.tsv")
__FILEPATH_FRESH_DATA = os.path.join(
    __DATA_FOLDER, "a2_RestaurantReviews_FreshDump.tsv")


def main():

    # Check if filepath argument exists
    if len(sys.argv) != 2:
        print(
            "Invalid argument(s)! Please use: python [current_file_path.py] [data_file_path.tsv]")
        sys.exit(1)

    # Check if filepath argument is "historical" or "fresh" (for default data files)
    if sys.argv[1] == "historical":
        if not os.path.isfile(__FILEPATH_HISTORICAL_DATA):
            print("Invalid argument: historical data file does not exist.")
            sys.exit(1)
        preprocess(filepath=__FILEPATH_HISTORICAL_DATA)
    elif sys.argv[1] == "fresh":
        if not os.path.isfile(__FILEPATH_FRESH_DATA):
            print("Invalid argument: fresh data file does not exist.")
            sys.exit(1)
        preprocess(filepath=__FILEPATH_FRESH_DATA)

    # Check if filepath argument is a .tsv file
    elif not sys.argv[1].endswith(".tsv"):
        print("Invalid argument:",
              sys.argv[1], "is required to be a filepath to a .tsv file.")
        sys.exit(1)

    # Check if file exists
    elif not os.path.isfile(sys.argv[1]):
        print("Invalid argument:", sys.argv[1], "does not exist.")
        sys.exit(1)

    else:
        # Preprocess the data and write to output folder as .csv
        preprocess(filepath=sys.argv[1])


def preprocess(filepath: str):
    _get_dataframe(filepath).to_csv(
        os.path.join(__OUTPUT_FOLDER, "preprocessed_" + os.path.basename(filepath)), sep="\t", index=False
    )


def _get_dataframe(filepath: str) -> pd.Dataframe:
    dataset = pd.read_csv(filepath, delimiter='\t', quoting=3)
    return dataset['Review'].apply(__preprocess_review)


def __preprocess_review(review: str) -> str:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    return ' '.join(review)


if __file__ == "__main__":
    main()
