# Homework 1: Linear Regression and Neural Network Regression

The goal of this assignment was to predict cancer mortality rates based on tabular data, comparing the performance of a
linear regression model vs multi-layer perceptron models of varying widths and depths.

The dataset was provided as a [csv file](./data/cancer_reg.csv), which we had to preprocess and split into training, test, and
validation datasets before training. We then had to implement the models and training scripts.

## Install Dependencies

Install `pipenv` if you don't already have it:
```
pip install pipenv --user
```

Navigate to the root of the submission folder and run:
```
pipenv install
```

## Test the Models

To test the linear regression model run:
```
pipenv run python3 ./test_model_linreg.py
```

To test the ANN model run:
```
pipenv run python3 ./test_model_ann.py
```

## Alternative to pipenv
If you don't wish to use `pipenv` a requirements.txt file has been provided. You can create a virtual environment with `venv` and use `pip` and the requirements.txt to install the required dependencies. At that point you should be able to run the test_model_*.py scripts directly, without using `pipenv run`.
