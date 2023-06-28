import os
import random
import sys
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
    train, test = train_test_split(setup_training, test_size = 0.2)
    prediction_result = production_phase.predict_sentiment(test, verbose=True) 
    counter = 0
    correct = 0
    incorrect = 0
    for t in test["Liked"]:
        if (t == prediction_result[counter]):
            correct+=1
        else:
            incorrect+=1
        counter += 1
    print("accuracy: " + str(correct / len(test))) 
    assert (correct / len(test) >= 0.85)
