# Sentiment Analysis 🎲

[![Latest Tag](https://img.shields.io/github/tag/remla23-team19/model-training.svg)](https://github.com/remla23-team19/model-training/tags) [![Latest Commit](https://img.shields.io/github/last-commit/remla23-team19/model-training.svg)](https://github.com/remla23-team19/model-training/commits/main)

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
pip install -r requirements.txt
```

> Note: if you are using Windows, replace `source venv/bin/activate` with `venv\Scripts\activate`

### Credits

This project is based on Skillcate AI "Sentiment Analysis Project — with traditional ML & NLP".
