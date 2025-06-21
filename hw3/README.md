# Homework 3: Medical Image Segmentation

The goal of this assignment was to perform image segmentation on medical images. The
Retina Blood Vessel Segmentation dataset used where the objective is to segment
retina blood vessels from retinal fundus images.

We used a U-Net architecture to perform the segmentation and explored the effects of
adding more layers to the U-Net, performing batch normalization, and adjusting hyperparameters.

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
