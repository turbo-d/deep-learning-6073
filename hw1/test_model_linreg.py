import pandas as pd
from tensorflow import keras

test = pd.read_csv("./test.csv", encoding="ISO-8859-1")
X_test = test.drop(columns=['TARGET_deathRate'])
y_test = test['TARGET_deathRate']

model = keras.saving.load_model("LinearRegression_0.01")

model.evaluate(X_test, y_test, batch_size=32, verbose=1)