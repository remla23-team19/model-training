# Sentiment Analysis 🎲

[![Latest Tag](https://img.shields.io/github/tag/remla23-team19/model-training.svg)](https://github.com/remla23-team19/model-training/tags) [![Latest Commit](https://img.shields.io/github/last-commit/remla23-team19/model-training.svg)](https://github.com/remla23-team19/model-training/commits/main) [![Python Version](https://img.shields.io/badge/python-3.8-yellow.svg)](https://www.python.org/downloads/release/python-380/) [![Code Quality CI](https://github.com/remla23-team19/model-training/actions/workflows/ci.yaml/badge.svg)](https://github.com/remla23-team19/model-training/actions/workflows/ci.yaml)

This is a project to train a model that performs sentiment analysis on restaurant reviews.
The training pipeline is in `b1_Sentiment_Analysis_Model.ipynb`.
The inference pipeline is in `b2_Sentiment_Predictor.ipynb`.
Training data is in `a1_RestaurantReviews_HistoricDump.tsv`.

## Structure

- `/data`: contains the training data
- `/models`: contains trained models
- `/scripts`: contains all pieces of the pipeline
- `/backup`: contains old version of the repository, for reference only

## Instructions

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

## Pipeline
Disclaimer, follow in chronological order to reduce problems with missing data/output. The pipeline is designed to be run from the root directory of the repository. You can run the pipeline using the following command:
```sh
dvc repro
```
As for the details, the model-training pipeline consists of the following phases:

### Data Collection 🗄️
The data is collected from Google Drive (remote storage) and can be loaded using:
```sh
python3 scripts/data_phase.py
```

Alternatively, it can also be loaded using `dvc`:
```sh
dvc pull
```

However, this requires you to authenticate with Google Drive and have access to the shared folder.
Therefore, the first option is used primarily and the second option is only used to demonstrate the use of `dvc` and best practices.

### Preprocessing 🚜
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

### Training 🥷
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

These can be used in the production phase of the pipeline. In fact, every time you add a version tag (e.g. v1.0.0) to the repository, the models will be automatically stored in the release. With this versioning system, you can easily track which models were used for which release and reproduce the results.

Furthermore, the performance metrics (confusion matrix and accuracy) will be stored in `/output` with the filename `performance_metrics_naive_bayes_model.json` and printed by default.

Note, using `dvc` you can check if any changes in the experiment yield different metrics as follows:
* Make a change, e.g. put `test_size=0.30` instead of `test_size=0.20` in `scripts/training_phase.py`
* Run `dvc exp run` to reproduce the experiment
* Run `dvc metrics diff` to compare the metrics, for this example it should yield:
* 
![image](https://github.com/remla23-team19/model-training/assets/56686692/6f0ca7f6-fa97-4fc9-80f2-b5fb3024e7e2)


### Production 🚀
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
To improve the code quality as much as possible, the following tools have been utilised to adhere to best practices. Note, the following tools are also used in the CI/CD pipeline and reports are generated automatically.

### PyLint
Run via `pylint ./scripts`. The following output should be observed:

![image](https://github.com/remla23-team19/model-training/assets/56686692/da07a177-f39f-4a94-beef-dfffe6414bf1)

> Note: PyLint is configured such that DSLinter is also automatically run!

### MLLint
Run via `mllint run`. The following output regarding (data) version control should be observed:

![image](https://github.com/remla23-team19/model-training/assets/56686692/9f847ba0-99ff-4660-9b6b-f9cb883f2559)


## Credits

This project is based on Skillcate AI "Sentiment Analysis Project — with traditional ML & NLP".
