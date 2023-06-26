

import os
import pickle
import sys
from typing import Union
import joblib

import pandas as pd

# Define folder locations
__ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
__DATA_FOLDER = os.path.join(__ROOT_FOLDER, "data")
__OUTPUT_FOLDER = os.path.join(__ROOT_FOLDER, "output")


# Define default datafile locations
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

    filepath: str = sys.argv[1]

    if filepath == "historical":
        filepath = __FILEPATH_HISTORICAL_DATA
        if not os.path.exists(filepath):
            print("Invalid argument: the default historical datafile (" +
                  str(filepath) + ") does not exist.")
            sys.exit(1)
    elif filepath == "fresh":
        filepath = __FILEPATH_FRESH_DATA
        if not os.path.exists(filepath):
            print("Invalid argument: the default fresh datafile (" +
                  str(filepath) + ") does not exist.")
            sys.exit(1)

    # Check if filepath argument is a(n existing) .tsv file
    if not filepath.endswith(".tsv"):
        print("Invalid argument:",
              filepath, "is required to be a filepath to a .tsv file.")
        sys.exit(1)

    if not os.path.isfile(filepath):
        print("Invalid argument:", filepath, "does not exist.")
        sys.exit(1)

    # Load and predict sentiment data
    data = pd.read_csv(filepath,
                       delimiter='\t',
                       quoting=3,
                       dtype={'Review': object, 'Liked': int})[:]

    predict_sentiment(data, verbose=True)


def predict_sentiment(input: Union[str, pd.DataFrame], model: str = 'c2_Classifier_Sentiment_Model', bow: str = 'c1_BoW_Sentiment_Model.pkl', verbose: bool = True):

    # Check if model exists
    model_path = os.path.join(__OUTPUT_FOLDER, model)
    if not os.path.isfile(model_path):
        print("Invalid argument:", model, "does not exist.")
        sys.exit(1)

    # Check if bag of words exists
    bow_path = os.path.join(__OUTPUT_FOLDER, bow)
    if not os.path.isfile(bow_path):
        print("Invalid argument:", bow, "does not exist.")
        sys.exit(1)

    # Load model and bag of words
    classifier = joblib.load(model_path)
    with open(bow_path, 'rb') as f:
        cv = pickle.load(f)

    # Load input (either string or dataframe)
    if isinstance(input, str):
        data = pd.DataFrame([input], columns=["Review"])
    elif isinstance(input, pd.DataFrame):
        data = input["Review"]
    else:
        print("Invalid argument: input must be of type str or pandas.DataFrame")
        sys.exit(1)

    X = cv.transform(data).toarray()
    y_pred = classifier.predict(X)

    prediction_map = {
        0: "negative",
        1: "positive"
    }

    if verbose:

        print("""
##############################
# SENTIMENT ANALYSIS RESULTS #
##############################
        """)

        for i in range(len(data)):
            print(data[i], ":", str(y_pred[i]) + " (" +
                  str(prediction_map[y_pred[i]]) + ")")

    return y_pred


if __name__ == "__main__":
    main()
