# credit-fraud-api

This is a personal project centred around building a simple, scalable model to predict occurrences of credit card fraud.

Multiple versions of the model will be made accessible via an API hosted using AWS Lambda.

The train and test datasets can be found on Kaggle here: https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv

# Development

For development, create a virtual environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run tests with pytest, run

```
pytest
```