"""
Create the ML models by running part of the pipeline
"""
from scripts.data_phase import load_data
from scripts.preprocessing_phase import FILEPATH_HISTORICAL_DATA, preprocess
from scripts.training_phase import FILEPATH_PREPROCESSED_HISTORICAL_DATA, train


def create_models():
    """
    This function creates the machine learning models by running part of
    the pipeline (preprocessing and training) with the default historical
    datafile. It is used to create the models for the production phase.
    """
    load_data()
    preprocess(FILEPATH_HISTORICAL_DATA)
    train(FILEPATH_PREPROCESSED_HISTORICAL_DATA)


if __name__ == "__main__":
    create_models()
