"""
PyTest for checking the robustness of the model.
"""
import os
import random
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))


from scripts import data_phase, preprocessing_phase, training_phase


@pytest.fixture(name="setup_training")
def fixture_setup_training():
    """
    Fixture for setting up the training environment.

    Returns:
        pd.DataFrame: The preprocessed historical data used for training.

    """
    data_phase.load_data()
    return preprocessing_phase.preprocess(preprocessing_phase.FILEPATH_HISTORICAL_DATA)


def test_robustness(setup_training):
    """
    Check if the model is robust to non-determinism by comparing the accuracy of the model
    when using the same seed and when using a random seed.

    Args:
        setup_training (pd.DataFrame): The preprocessed historical data used for training.

    """
    random.seed(0)
    _, base_accuracy = training_phase.naive_bayes_model(
        setup_training, seed=0, verbose=True
    )

    for _ in range(5):
        random_seed = random.randint(0, 1000000)
        _, random_accuracy = training_phase.naive_bayes_model(
            setup_training, seed=random_seed, verbose=True
        )
        assert abs(base_accuracy - random_accuracy) < 0.1
