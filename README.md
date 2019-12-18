# Kaggle-House-Prices
[House prices predictor](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  


## Visualizing and analyzing
Below is a heatmap, which is a matrix that describes the correlation between all the different numerical parameters.  

![heatmap](https://user-images.githubusercontent.com/45593399/70901731-a8f37000-1ffb-11ea-94a7-1a2d31cc09c4.png)  

It is easy to see somefeatures that are directly correlated with the selling prize, such as the "Overall quality" feature with a correlation score of 0.8, or "General living area" as plotted below.  

![housing price](https://user-images.githubusercontent.com/45593399/70901739-abee6080-1ffb-11ea-853d-370e8d40b3eb.PNG)  

One could potentially get a good estimator by only using the few most correlated features (Mean Square Error of about 1.5-4), but then you would also loose a lot of the information that sits in the data, so the approach i will be doing is running the normalized features in to a Neural Network for it to learn more complex patterns.

## Feature engineering
The feature "MSSubClass" does not seem to be a numerical feature, so it will be made to an categorical feature
```
houses['MSSubClass'] = houses['MSSubClass'].astype(str)
```
Afterwards all categorical features were one hot encoded to be able to gather the information within the features. Below is an example on what one hot encoding is, read more on it [here!](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
![one hot encoding](https://user-images.githubusercontent.com/45593399/71045856-d133b980-2136-11ea-954f-fde154fb5ea9.PNG)  

All of the missing values were filled in with the mean value of the collumn. This could and SHOULD be given more thought in a more important setting, different features will have different meaning behind the missing values, either "None" to mean "0" or "Not Aviable" to mean some unknown value we can approximate to be the mean of the data we know. Or we could try to impute the missing values in other ways.( other predicted or representative values are filled in place of the missing data.)  

For normalization a MinMaxScaler was fitted for both the training and testing set, and then each set was separatly transformed so that both would have the same relative transformation. Below is a snippet of code
```
houses = pd.concat([train_data,test_data], sort=False)

scaler_features = scaler_features.fit(houses.values)  # Fit the scaler to the combined test and training set
train_data = houses[:len(train_data.index)]
test_data = houses[len(train_data.index):]

test_data = pd.DataFrame(scaler_features.transform(test_data.values), columns=test_data.columns)
train_data = pd.DataFrame(scaler_features.transform(train_data.values), columns=train_data.columns)
```

## Results
As mentioned above, if we only look at one or two features we can quickly get a MSE = 2. But with the feature engineering from above, this can greatly be reduced to consistently below MSE = 0.2!  
I have tried different models with between two and five hidden layers, a hidden node count between 30 and 1000 and a range of epochs between 100 to 5000. Optmila results for my setup came with two hidden layers with 300 units each, and using relu activation function and 500 epochs with batch size 32. This gave a score = 0.15427, putting me in the 65th percentile in this competition.  
![score](https://user-images.githubusercontent.com/45593399/71048505-be71b280-213f-11ea-9ba1-78be8cf1c53e.PNG)  

But by looking on the leaderboards on Kaggle it is reasonable to believe that more time spent on feature engineering, or using another statistical model(mentioned in improvements) could realisticly lower this MSE down to around 1.2-1.3 (25th percentile).  

## Improvements
The top scorers has had great success with the lasso regression, which does a good job of avoiding overfitting. Other regression models which seems to work well are ridge regression, forest regression and xgboost regression. User Luca Basanisi has a [great notebook](https://www.kaggle.com/lucabasa/houseprice-end-to-end-project/notebook) on Kaggle where he goes much more in depth on the different models and how to better analyze and preprocess the data.  
Since the features in this dataset are really intuitive for humans, it would be reconmended to group certain highly correlated features together, which could lead to less false confidence since it no longer trains on two features which give similar output.  
![housing](https://user-images.githubusercontent.com/45593399/71048894-152bbc00-2141-11ea-9189-39a607b02a0e.png)
