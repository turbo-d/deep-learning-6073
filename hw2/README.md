# Homework 2: Convolutional Neural Network

The goal of this assignment was to classify handwritten digits, using the MNIST dataset. We compared
the performance across three different architectures:
- Multilayer perceptron
- LeNet-5
- ResNet18
as well as different choices of hyperparameters.


## Install Dependencies

Create a virtual environment with `venv`. Install the requirements from
the requirements.txt file:
```
pip install -r requirements.txt
```

## Test the Models
The eval.py script with load the test dataset, load the specified model,
and output both the test results and a feature visualization of the first
two layers in the CNN models.

To test the best performing dnn model run:
```
pipenv run python3 ./eval.py dnn --lr 0.001
```
To test the best performing lenet model run:
```
pipenv run python3 ./eval.py lenet --lr 0.001
```
To test the best performing resnet model run:
```
pipenv run python3 ./eval.py resnet --lr 0.0001
```
