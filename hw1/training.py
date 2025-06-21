import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

# Import data sets
train = pd.read_csv("./train.csv", encoding="ISO-8859-1")
X_train = train.drop(columns=['TARGET_deathRate'])
y_train = train['TARGET_deathRate']

val = pd.read_csv("./validation.csv", encoding="ISO-8859-1")
X_val = val.drop(columns=['TARGET_deathRate'])
y_val = val['TARGET_deathRate']

test = pd.read_csv("./test.csv", encoding="ISO-8859-1")
X_test = test.drop(columns=['TARGET_deathRate'])
y_test = test['TARGET_deathRate']

# Define models
def LinearRegression(n_feat, learning_rate):
  model = keras.models.Sequential([
    keras.Input(shape=(n_feat,)),
    keras.layers.Dense(1, use_bias=True)
  ])

  mse = keras.losses.MeanSquaredError()
  opt = keras.optimizers.legacy.SGD(learning_rate=learning_rate)
  model.compile(loss=mse, optimizer=opt)

  return model

def ANN_oneL_16(n_feat, learning_rate):
  model = keras.models.Sequential([
    keras.Input(shape=(n_feat,)),
    keras.layers.Dense(16, activation='relu', use_bias=True),
    keras.layers.Dense(1)
  ])

  mse = keras.losses.MeanSquaredError()
  opt = keras.optimizers.legacy.SGD(learning_rate=learning_rate)
  model.compile(loss=mse, optimizer=opt)

  return model

def ANN_twoL_32_8(n_feat, learning_rate):
  model = keras.models.Sequential([
    keras.Input(shape=(n_feat,)),
    keras.layers.Dense(32, activation='relu', use_bias=True),
    keras.layers.Dense(8, activation='relu', use_bias=True),
    keras.layers.Dense(1)
  ])

  mse = keras.losses.MeanSquaredError()
  opt = keras.optimizers.legacy.SGD(learning_rate=learning_rate)
  model.compile(loss=mse, optimizer=opt)

  return model

def ANN_threeL_32_16_8(n_feat, learning_rate):
  model = keras.models.Sequential([
    keras.Input(shape=(n_feat,)),
    keras.layers.Dense(32, activation='relu', use_bias=True),
    keras.layers.Dense(16, activation='relu', use_bias=True),
    keras.layers.Dense(8, activation='relu', use_bias=True),
    keras.layers.Dense(1)
  ])

  mse = keras.losses.MeanSquaredError()
  opt = keras.optimizers.legacy.SGD(learning_rate=learning_rate)
  model.compile(loss=mse, optimizer=opt)

  return model

def ANN_fourL_32_16_8_4(n_feat, learning_rate):
  model = keras.models.Sequential([
    keras.Input(shape=(n_feat,)),
    keras.layers.Dense(32, activation='relu', use_bias=True),
    keras.layers.Dense(16, activation='relu', use_bias=True),
    keras.layers.Dense(8, activation='relu', use_bias=True),
    keras.layers.Dense(4, activation='relu', use_bias=True),
    keras.layers.Dense(1)
  ])

  mse = keras.losses.MeanSquaredError()
  opt = keras.optimizers.legacy.SGD(learning_rate=learning_rate)
  model.compile(loss=mse, optimizer=opt)

  return model

# Train each model with each of the prescribed learning rates, then evaluate the model

# Linear Regression
for lr in [0.1, 0.01, 0.001, 0.0001]:
  model_name = "LinearRegression"+"_"+str(lr)
  print()
  print(model_name)

  model = LinearRegression(X_train.shape[1], lr)
  history = model.fit(X_train, y_train, batch_size=32, epochs=150, verbose=1, validation_data=(X_val, y_val))

  test_loss = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

  keras.saving.save_model(model, model_name, overwrite=True, save_format="tf")

  fig, ax = plt.subplots()
  ax.plot(history.history['loss'])
  ax.plot(history.history['val_loss'])
  ax.set_title(model_name + ' loss')
  ax.set_ylabel('loss')
  ax.set_xlabel('epoch')
  ax.legend(['train', 'val'], loc='upper left')
  ax.text(0.5, 0.9, "test loss: " + str(test_loss), transform=ax.transAxes)
  fig.savefig(model_name + '.png')

# ANN-oneL-16
for lr in [0.1, 0.01, 0.001, 0.0001]:
  model_name = "ANN-oneL-16"+"_"+str(lr)
  print()
  print(model_name)

  model = ANN_oneL_16(X_train.shape[1], lr)
  history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val))

  test_loss = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

  keras.saving.save_model(model, model_name, overwrite=True, save_format="tf")

  fig, ax = plt.subplots()
  ax.plot(history.history['loss'])
  ax.plot(history.history['val_loss'])
  ax.set_title(model_name + ' loss')
  ax.set_ylabel('loss')
  ax.set_xlabel('epoch')
  ax.legend(['train', 'val'], loc='upper left')
  ax.text(0.5, 0.9, "test loss: " + str(test_loss), transform=ax.transAxes)
  fig.savefig(model_name + '.png')

# ANN-twoL-32-8
for lr in [0.1, 0.01, 0.001, 0.0001]:
  model_name = "ANN-twoL-32-8"+"_"+str(lr)
  print()
  print(model_name)

  model = ANN_twoL_32_8(X_train.shape[1], lr)
  history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val))

  test_loss = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

  keras.saving.save_model(model, model_name, overwrite=True, save_format="tf")

  fig, ax = plt.subplots()
  ax.plot(history.history['loss'])
  ax.plot(history.history['val_loss'])
  ax.set_title(model_name + ' loss')
  ax.set_ylabel('loss')
  ax.set_xlabel('epoch')
  ax.legend(['train', 'val'], loc='upper left')
  ax.text(0.5, 0.9, "test loss: " + str(test_loss), transform=ax.transAxes)
  fig.savefig(model_name + '.png')

# ANN-threeL-32-16-8
for lr in [0.1, 0.01, 0.001, 0.0001]:
  model_name = "ANN-threeL-32-16-8"+"_"+str(lr)
  print()
  print(model_name)

  model = ANN_threeL_32_16_8(X_train.shape[1], lr)
  history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val))

  test_loss = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

  keras.saving.save_model(model, model_name, overwrite=True, save_format="tf")

  fig, ax = plt.subplots()
  ax.plot(history.history['loss'])
  ax.plot(history.history['val_loss'])
  ax.set_title(model_name + ' loss')
  ax.set_ylabel('loss')
  ax.set_xlabel('epoch')
  ax.legend(['train', 'val'], loc='upper left')
  ax.text(0.5, 0.9, "test loss: " + str(test_loss), transform=ax.transAxes)
  fig.savefig(model_name + '.png')

# ANN-fourL-32-16-8-4
for lr in [0.1, 0.01, 0.001, 0.0001]:
  model_name = "ANN-fourL-32-16-8-4_"+str(lr)
  print()
  print(model_name)

  model = ANN_fourL_32_16_8_4(X_train.shape[1], lr)
  history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val))

  test_loss = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

  keras.saving.save_model(model, model_name, overwrite=True, save_format="tf")

  fig, ax = plt.subplots()
  ax.plot(history.history['loss'])
  ax.plot(history.history['val_loss'])
  ax.set_title(model_name + ' loss')
  ax.set_ylabel('loss')
  ax.set_xlabel('epoch')
  ax.legend(['train', 'val'], loc='upper left')
  ax.text(0.5, 0.9, "test loss: " + str(test_loss), transform=ax.transAxes)
  fig.savefig(model_name + '.png')
