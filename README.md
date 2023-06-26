# Sentiment Analysis 🎲

[![Latest Tag](https://img.shields.io/github/tag/remla23-team19/model-training.svg)](https://github.com/remla23-team19/model-training/tags) [![Latest Commit](https://img.shields.io/github/last-commit/remla23-team19/model-training.svg)](https://github.com/remla23-team19/model-training/commits/main) [![Python Version](https://img.shields.io/badge/python-3.8-yellow.svg)](https://www.python.org/downloads/release/python-380/)

This is a project to train a model that performs sentiment analysis on restaurant reviews.
The training pipeline is in `b1_Sentiment_Analysis_Model.ipynb`.
The inference pipeline is in `b2_Sentiment_Predictor.ipynb`.
Training data is in `a1_RestaurantReviews_HistoricDump.tsv`.

### Structure

- `/data`: contains the training data
- `/models`: contains trained models
- `/scripts`: contains all pieces of the pipeline
- `/backup`: contains old version of the repository, for reference only

### Instructions

Clone the repository:

```sh
git clone https://github.com/remla23-team19/model-training.git
```

Install dependencies using a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-pip.txt
```

> Note: if you are using Windows, replace `source venv/bin/activate` with `venv\Scripts\activate`

Alternatively, use `pyenv` and `pipenv`:

```sh
pyenv install 3.8
pyenv global 3.8
pipenv --python /Users/username/.pyenv/shims/python
pipenv install
```

> Note: the requirements-pipenv.txt file can also be used to install the dependencies using pip.

Get the data via `dvc`:

```sh
cd data
dvc pull
```

> Note: this will require you to authenticate with Google Drive, and you will need to have access to the shared folder. Therefore, the data is also available in the `data` folder. This is not normally the case (especially with large files), but we are doing this for the sake of the project to demonstrate the use of `dvc` and best practices.


### Pipeline
Disclaimer, follow in chronological order to reduce problems with missing data/output.
The model-training pipeline consists of the following phases:

#### Preprocessing
Preprocess the `/data` using `/scripts/preprocessing_phase.py`:
```sh
python3 [current_file_path.py] [data_file_path.tsv]
```

To simplify, the alias 'historical' can also be used to achieve the same as:
```sh
python3 scripts/preprocessing_phase.py historical
python3 scripts/preprocessing_phase.py data/a1_RestaurantReviews_HistoricDump.tsv 
```

Running this script will result in a preprocessed file stored in `/output` with the filename `preprocessed_[data_file_path.tsv]`.

#### Training
Train the model using `/scripts/training_phase.py`:
```sh
python3 [current_file_path.py] [data_file_path.tsv]
```

To simplify, the alias 'historical' can also be used to achieve the same as:
```sh
python3 scripts/training_phase.py historical
python3 scripts/training_phase.py output/preprocessed_a1_RestaurantReviews_HistoricDump.tsv
```

Running this script will result in two stored models in `/models`:
* Bag of Words (BoW) model: `c1_BoW_Sentiment_Model.pkl`
* Classifier model: `c2_Classifier_Sentiment_Model`

These can be used in the production phase of the pipeline.

Furthermore, the performance metrics (confusion matrix and accuracy) will be stored in `/output` with the filename `performance_metrics_naive_bayes_model.json` and printed by default.

#### Production
Run the production phase using `/scripts/production_phase.py`:
```sh
python3 [current_file_path.py] [data_file_path.tsv]
```

To simplify, the alias 'historical' and/or 'fresh' can also be used to achieve the same as:
```sh
python3 scripts/production_phase.py historical
python3 scripts/production_phase.py data/a1_RestaurantReviews_HistoricDump.tsv

python3 scripts/production_phase.py fresh
python3 scripts/production_phase.py data/a2_RestaurantReviews_FreshDump.tsv
```

Running this script will print and return the predicted sentiment of the reviews in the given data file. By default, the script will use the BoW and Classifier model from the training phase. If you want to use a different model, please update the models in `/models` and change the model names in the script.

## Code Quality
To improve the code quality as much as possible, the following tools have been utilised to adhere to best practices.

### PyLint
Run via `pylint ./scripts`. The following output should be observed:

![image](https://github.com/remla23-team19/model-training/assets/56686692/da07a177-f39f-4a94-beef-dfffe6414bf1)

> Note: PyLint is configured such that DSLinter is also automatically run!

### MLLint
Run via `mllint run`. The following output regarding (data) version control should be observed:

![image](https://github.com/remla23-team19/model-training/assets/56686692/9f847ba0-99ff-4660-9b6b-f9cb883f2559)


## Credits

This project is based on Skillcate AI "Sentiment Analysis Project — with traditional ML & NLP".
