import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import  MinMaxScaler
scaler_features = MinMaxScaler()
scaler_predict = MinMaxScaler()

# Training data
train_data = pd.read_csv("train.csv")
train_data = train_data.drop("Id", axis=1)
train_target = train_data["SalePrice"]
train_data = train_data.drop("SalePrice", axis=1)


# Testing data
test_data = pd.read_csv("test.csv")
test_id = test_data["Id"]
test_data = test_data.drop("Id", axis=1)


# Concatonate training and testing data before one hot encoding the categorical features
houses = pd.concat([train_data,test_data], sort=False)

# Make MSSubclass to a katecategorical feature instead of a numerical
houses['MSSubClass'] = houses['MSSubClass'].astype(str)

# one hot encoding, and split data afterwards, end up with 303 columns
houses = pd.get_dummies(houses)
houses = houses.fillna(houses.mean())
scaler_features = scaler_features.fit(houses.values)  # Fit the scaler to the combined test and training set
train_data = houses[:len(train_data.index)]
test_data = houses[len(train_data.index):]




# Normalize the  feature data with the feature scaler
test_data = pd.DataFrame(scaler_features.transform(test_data.values), columns=test_data.columns)
train_data = pd.DataFrame(scaler_features.transform(train_data.values), columns=train_data.columns)
train_target = pd.DataFrame(scaler_predict.fit_transform(train_target.values.reshape(-1,1)), columns=["SalePrice"]) # Reshape is necessary on single columns

x = train_data
y = train_target

test_x = test_data


"""Make model and add hidden layers and softmax at the end"""
hidden = 303
def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=hidden, input_dim=len(x.columns), activation="relu"))
    model.add(tf.keras.layers.Dense(units=hidden, activation="relu"))
    model.add(tf.keras.layers.Dense(units=hidden, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae','accuracy'])
    return model

from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_model, batch_size=32,epochs=500)

results=regressor.fit(x,y)

y_pred= regressor.predict(x)

fig, ax = plt.subplots()
ax.scatter(y, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')


MSE = np.mean(np.square(np.subtract(np.asarray(y_pred), np.asarray(y))))
print("MSE on training data is: ", MSE)

plt.show()

"""Save predictions on test houses"""
test_pred = regressor.predict(test_x)
test_pred = test_pred.reshape(-1,1)
output = pd.DataFrame(test_id,columns=["Id"])
output["SalePrice"] = scaler_predict.inverse_transform(test_pred)
output.to_csv(r'C:\Users\Joule\Desktop\export_dataframe.csv', index = None)