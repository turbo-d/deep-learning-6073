# Homework 4: Machine Translator

The goal of this assignment was to perform machine translation from
German to English using the Multi30K dataset, in order to better understand
sequence to sequence models.

The architecture used was a sequence-to-sequence transformer, with a pretrained
model used for tokenization.

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

To test the best performing model run:
```
pipenv run python3 ./eval.py
```

## Alternative to pipenv
If you don't wish to use `pipenv` a requirements.txt file has been provided. You can create a virtual environment with `venv` and use `pip` and the requirements.txt to install the required dependencies. At that point you should be able to run the eval.py script directly, without using `pipenv run`.
