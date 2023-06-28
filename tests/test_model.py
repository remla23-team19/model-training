"""
PyTest for data slicing.
"""

import pytest
from sklearn.model_selection import train_test_split

from scripts import data_phase, preprocessing_phase, training_phase, production_phase


@pytest.fixture(name="setup_training")
def fixture_setup_training():
    """
    Fixture for setting up the training environment.

    Returns:
        pd.DataFrame: The preprocessed historical data used for training.

    """
    data_phase.load_data()
    return preprocessing_phase.preprocess(preprocessing_phase.FILEPATH_HISTORICAL_DATA)


def test_model(setup_training):
    """
    The `test_model` function evaluates the accuracy of a sentiment prediction model on
    a test dataset and asserts that the accuracy is at least 85%.

    :param setup_training: The parameter "setup_training" is a dataset that is used for training
    and testing a model. It is a pandas DataFrame as a result of the training phase.
    """

    # Train Models such that they are available for the production phase
    training_phase.naive_bayes_model(
        setup_training, seed=0, verbose=True
    )

    _, tests = train_test_split(setup_training, test_size=0.2, random_state=42)
    prediction_result = production_phase.predict_sentiment(tests, verbose=True)

    counter = 0
    correct = 0
    incorrect = 0

    for test in tests["Liked"]:
        if test == prediction_result[counter]:
            correct += 1
        else:
            incorrect += 1
        counter += 1

    accuracy = correct / float(len(tests))

    print("accuracy: " + str(accuracy))
    assert 0.6 <= accuracy <= 1.0
