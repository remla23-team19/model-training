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


def main():
    """
    This function checks if the input argument is valid and calls the preprocess function with the input
    file path.
    """

    # Check if filepath argument exists
    if len(sys.argv) != 2:
        print(
            "Invalid argument(s)! Please use: python [current_file_path.py] [data_file_path.tsv]")
        sys.exit(1)

    # Check if filepath argument is a .tsv file
    if not sys.argv[1].endswith(".tsv"):
        print("Invalid argument:",
              sys.argv[1], "is required to be a filepath to a .tsv file.")
        sys.exit(1)

    # Check if file exists
    elif not os.path.isfile(sys.argv[1]):
        print("Invalid argument:", sys.argv[1], "does not exist.")
        sys.exit(1)

    # Preprocess the data and write to output folder as .csv
    preprocess(filepath=sys.argv[1])


def preprocess(filepath: str):
    """
    The function preprocesses data from a file and saves it as a CSV file in an output folder.

    :param filepath: The filepath parameter is a string that represents the path to the input file that
    needs to be preprocessed
    :type filepath: str
    """

    # Create output folder if it does not exist
    if not os.path.exists(__OUTPUT_FOLDER):
        os.makedirs(__OUTPUT_FOLDER)

    # Preprocess the data and write to output folder as .csv
    _get_dataframe(filepath).to_csv(
        os.path.join(__OUTPUT_FOLDER, "preprocessed_" + os.path.basename(filepath)), sep="\t", index=False
    )


def _get_dataframe(filepath: str) -> pd.DataFrame:
    """
    The function reads a CSV file and returns a preprocessed column of the dataset.

    :param filepath: The `filepath` parameter is a string that represents the path to the file
    containing the dataset to be read
    :type filepath: str
    :return: a pandas DataFrame object after reading a CSV file from the specified filepath and applying
    a preprocessing function called "__preprocess_review" to the "Review" column of the dataset.
    """
    # Read from CSV
    data: pd.DataFrame = pd.read_csv(filepath, delimiter='\t', quoting=3, dtype={
        'Review': object, 'Liked': int})[:]

    # Preprocess the "Review" column
    data['Review'] = data['Review'].apply(__preprocess_review)

    # Return the preprocessed DataFrame
    return data


def __preprocess_review(review: str) -> str:
    """
    This function preprocesses a given review by removing non-alphabetic characters, converting it to
    lowercase, splitting it into words, stemming each word, and removing stopwords before returning the
    preprocessed review as a string.

    :param review: A string representing a review that needs to be preprocessed
    :type review: str
    :return: Preprocessed version of the input `review` string. 
    """
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    return ' '.join(review)


if __name__ == "__main__":
    print("Preprocessing data...")
    main()
