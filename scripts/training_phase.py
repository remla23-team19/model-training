import json
import os
import pickle
import sys
from typing import Tuple
import joblib
from numpy import ndarray
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Define folder locations
__ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
__MODEL_FOLDER = os.path.join(__ROOT_FOLDER, "models")
__OUTPUT_FOLDER = os.path.join(__ROOT_FOLDER, "output")

# Define default training datafile locations
__FILEPATH_PREPROCESSED_HISTORICAL_DATA = os.path.join(
    __OUTPUT_FOLDER, "preprocessed_a1_RestaurantReviews_HistoricDump.tsv")


def main():
    # Check if filepath argument exists
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Invalid argument(s)! Please use: python [current_file_path.py] [data_file_path.tsv] [optional: seed int]")
        sys.exit(1)

    # Check seed argument or set to default
    seed: int = 0
    if len(sys.argv) == 3:
        if not sys.argv[2].isdigit():
            print("Invalid argument: seed is required to be an integer.")
            sys.exit(1)
        else:
            seed = int(sys.argv[2])

    # Check filepath argument and aliases
    filepath: str = sys.argv[1]

    # Check if filepath is alias for a default training datafile
    if filepath == "historical":
        filepath = __FILEPATH_PREPROCESSED_HISTORICAL_DATA
        if not os.path.exists(filepath):
            print("Invalid argument: preprocessed historical training datafile (" +
                  str(filepath) + ") does not exist.")
            sys.exit(1)

    # Check if filepath argument is a .tsv file
    elif not filepath.endswith(".tsv"):
        print("Invalid argument:",
              sys.argv[1], "is required to be a filepath to a .tsv file.")
        sys.exit(1)

    # Check if file exists
    if not os.path.isfile(filepath):
        print("Invalid argument:", filepath, "does not exist.")
        sys.exit(1)

    # Train the model
    naive_bayes_model(pd.read_csv(filepath, delimiter='\t', dtype={
                      'Review': object, 'Liked': int})[:], seed, verbose=True)


def naive_bayes_model(input: pd.DataFrame, seed: int, verbose: bool = False) -> Tuple[ndarray, float]:
    """
    The function `naive_bayes_model` takes in a pandas DataFrame of text data, creates a Bag of Words
    (BoW) dictionary using CountVectorizer, trains a Gaussian Naive Bayes classifier on the data, and
    returns the confusion matrix and accuracy score of the model.

    :param input: The input parameter is a pandas DataFrame used training and testing the model. It
    should contain two columns: 'Review' and 'Liked'. The 'Review' column should contain the text data
    and the 'Liked' column should contain the labels (0 or 1).
    :type input: pd.DataFrame
    :param seed: The `seed` parameter is an integer value used to set the random seed for
    reproducibility of the results. It is used in the `train_test_split()` function to split the dataset
    into training and test sets. By setting the random seed, the same split can be obtained every time
    the function
    :type seed: int
    :param verbose: A boolean parameter that determines whether or not to print additional information
    during the execution of the function. If set to True, the function will print the confusion matrix
    and accuracy score. If set to False, the function will not print any additional information,
    defaults to False
    :type verbose: bool (optional)
    :return: a tuple containing the confusion matrix and accuracy score of a trained Gaussian Naive
    Bayes classifier model.
    """

    _MODEL_STORAGE_NAME: str = "c1_BoW_Sentiment_Model"

    if not os.path.isdir(__OUTPUT_FOLDER):
        os.mkdir(__OUTPUT_FOLDER)

    if not os.path.isdir(__MODEL_FOLDER):
        os.mkdir(__MODEL_FOLDER)

    # `cv = CountVectorizer(max_features = 1420)` is creating an instance of the CountVectorizer class
    # with a maximum number of features set to 1420. CountVectorizer is a method for converting text
    # data into a matrix of token counts, where each row represents a document and each column
    # represents a unique word in the corpus. The `max_features` parameter specifies the maximum
    # number of features to be extracted from the text data. In this case, the CountVectorizer will
    # only consider the top 1420 most frequent words in the corpus.
    cv = CountVectorizer(max_features=1420)

    # `X` is a matrix of features extracted from the text data in the 'Review' column of the input
    # DataFrame using the CountVectorizer method. The `fit_transform()` method fits the
    # CountVectorizer to the text data and transforms it into a matrix of token counts. The
    # `toarray()` method converts the sparse matrix into a dense matrix.
    X = cv.fit_transform(input['Review'].astype('U')).toarray()
    y = input.iloc[:, 1].values

    # Save the CountVectorizer object (Bag of Words (BoW) dictionary) as a pickle file
    with open(os.path.join(__MODEL_FOLDER, _MODEL_STORAGE_NAME + ".pkl"), "wb") as f:
        pickle.dump(cv, f)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed)

    # Create a Gaussian Naive Bayes classifier and train it on the training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Exporting NB Classifier to later use in prediction
    joblib.dump(classifier, os.path.join(
        __MODEL_FOLDER, 'c2_Classifier_Sentiment_Model'))

    # Predict the test set results
    y_pred = classifier.predict(X_test)

    # Evaluate the model using the confusion matrix and accuracy score
    cm = confusion_matrix(y_test, y_pred)
    acs = accuracy_score(y_test, y_pred)

    # Store performance metrics in a JSON file
    performance_metrics = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
        "accuracy": acs
    }

    if verbose:
        print("Performance Metrics:\n", performance_metrics)

    with open(os.path.join(__OUTPUT_FOLDER, 'performance_naive_bayes_model.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(performance_metrics))

    return (cm, acs)


if __name__ == "__main__":
    main()
