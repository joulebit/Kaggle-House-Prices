# Kaggle-House-Prices
[House prices predictor](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  


## Visualizing and analyzing
Below is a heatmap, which is a matrix that describes the correlation between all the different parameters.  

![heatmap](https://user-images.githubusercontent.com/45593399/70901731-a8f37000-1ffb-11ea-94a7-1a2d31cc09c4.png)  

It is easy to see somefeatures that are directly correlated with the selling prize, such as the "Overall quality" feature with a correlation score of 0.8, or "General living area" as plotted below.  

![housing price](https://user-images.githubusercontent.com/45593399/70901739-abee6080-1ffb-11ea-853d-370e8d40b3eb.PNG)  

One could potentially get a good estimator by only using the few most correlated features, but then you would also loose a lot of the information that sits in the data, so the approach i will be doing is running the normalized features in to a Neural Network for it to learn more complex patterns.

feature engineering
