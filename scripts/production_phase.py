"""
Production Phase of the Sentiment Analysis ML Pipeline
"""

import os
import pickle
import sys
from typing import List, Union

import joblib
import pandas as pd

# Define folder locations
__ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
__MODEL_FOLDER = os.path.join(__ROOT_FOLDER, "models")
__DATA_FOLDER = os.path.join(__ROOT_FOLDER, "data")
__OUTPUT_FOLDER = os.path.join(__ROOT_FOLDER, "output")


# Define default datafile locations
__FILEPATH_HISTORICAL_DATA = os.path.join(
    __DATA_FOLDER, "a1_RestaurantReviews_HistoricDump.tsv"
)
__FILEPATH_FRESH_DATA = os.path.join(
    __DATA_FOLDER, "a2_RestaurantReviews_FreshDump.tsv"
)


def main():
    """
    This function checks if a valid filepath argument exists, loads and predicts sentiment data
    from a .tsv file. It also allows for the default historical datafile to be used, if the
    input argument is "historical", and the default fresh datafile to be used, if the input
    argument is "fresh".
    """

    # Check if filepath argument exists
    if len(sys.argv) != 2:
        print(
            "Invalid argument(s)! Please use: python [current_file_path.py] [data_file_path.tsv]"
        )
        sys.exit(1)

    filepath: str = sys.argv[1]

    if filepath == "historical":
        filepath = __FILEPATH_HISTORICAL_DATA
        if not os.path.exists(filepath):
            print(
                "Invalid argument: the default historical datafile ("
                + str(filepath)
                + ") does not exist."
            )
            sys.exit(1)
    elif filepath == "fresh":
        filepath = __FILEPATH_FRESH_DATA
        if not os.path.exists(filepath):
            print(
                "Invalid argument: the default fresh datafile ("
                + str(filepath)
                + ") does not exist."
            )
            sys.exit(1)

    # Check if filepath argument is a(n existing) .tsv file
    if not filepath.endswith(".tsv"):
        print(
            "Invalid argument:",
            filepath,
            "is required to be a filepath to a .tsv file.",
        )
        sys.exit(1)

    if not os.path.isfile(filepath):
        print("Invalid argument:", filepath, "does not exist.")
        sys.exit(1)

    # Load and predict sentiment data
    data = pd.read_csv(
        filepath, delimiter="\t", quoting=3, dtype={"Review": object, "Liked": int}
    )[:]

    predict_sentiment(data, verbose=True)


def predict_sentiment(
    input_data: Union[str, pd.DataFrame],
    model: str = "c2_Classifier_Sentiment_Model",
    bow: str = "c1_BoW_Sentiment_Model.pkl",
    verbose: bool = True,
) -> List[int]:
    """
    This function takes in input data (either a string or a pandas DataFrame), a trained sentiment
    analysis model, and a bag of words model, and returns a list of predicted sentiment labels
    (0 for negative, 1 for positive) for each input data point.

    :param input_data: The input data for sentiment analysis. It can be either a string or a pandas
    DataFrame containing a column named "Review" with the text to analyze
    :type input_data: Union[str, pd.DataFrame]
    :param model: The name of the trained machine learning model to be used for sentiment analysis.
    Defaults to c2_Classifier_Sentiment_Model.
    :type model: str (optional)
    :param bow: bow stands for "bag of words" and refers to a pre-trained model that converts text
    data into numerical vectors that can be used as input for machine learning models. In this
    function, the bag of words model is loaded from a pickle file and used to transform the input
    data into numerical vectors. Defaults to c1_BoW_Sentiment_Model.pkl.
    :type bow: str (optional)
    :param verbose: A boolean parameter that determines whether or not to print the sentiment
    analysis results to the console. If set to True, the function will print the sentiment analysis
    results to the console. If set to False, nothing will be printed to the console.
    Defaults to True.
    :type verbose: bool (optional)
    :return: a list of integers representing the predicted sentiment of the input data.
    """

    # Check if model exists
    model_path = os.path.join(__MODEL_FOLDER, model)
    if not os.path.isfile(model_path):
        print("Invalid argument:", model, "does not exist.")
        sys.exit(1)

    # Check if bag of words exists
    bow_path = os.path.join(__MODEL_FOLDER, bow)
    if not os.path.isfile(bow_path):
        print("Invalid argument:", bow, "does not exist.")
        sys.exit(1)

    # Load model and bag of words
    classifier = joblib.load(model_path)
    with open(bow_path, "rb") as f:
        cv = pickle.load(f)

    # Load input data (either string or dataframe)
    if isinstance(input_data, str):
        data = pd.DataFrame([input_data], columns=["Review"])
    elif isinstance(input_data, pd.DataFrame):
        data = input_data["Review"]
    else:
        print("Invalid argument: input data must be of type str or pandas.DataFrame")
        sys.exit(1)

    X = cv.transform(data).toarray()
    y_pred = classifier.predict(X)

    prediction_map = {0: "negative", 1: "positive"}

    if verbose:
        print(
            """
##############################
# SENTIMENT ANALYSIS RESULTS #
##############################
        """
        )

        for i, item in enumerate(data):
            print(
                item, ":", str(y_pred[i]) + " (" + str(prediction_map[y_pred[i]]) + ")"
            )

    return y_pred


if __name__ == "__main__":
    main()
