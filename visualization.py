import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("train.csv")
del data["Id"]



#Normalize data within two standard deviations and find correlation scores

std_price = data["SalePrice"].std()
mean_price = data["SalePrice"].mean()

std_area = data["LotArea"].std()
mean_area = data["LotArea"].mean()


data = data[(data["SalePrice"] < 2*std_price + mean_price) & (data["SalePrice"] > -2*std_price + mean_price)]
data = data[(data["LotArea"] < 2*std_area + mean_area) & (data["LotArea"] > -2*std_area + mean_area)]


area = data["GrLivArea"]
sale_price = data["SalePrice"]

plt.subplot(1,2,1)


m,b = np.polyfit(area, sale_price, 1) 
plt.plot(area, sale_price, 'bo', area, m*area+b, 'k') 

plt.xlabel("General living area")
plt.ylabel("House selling price")

#plt.subplot(1,2,2)
#plt.xlabel("Correlation matrix")
print(data.corr()[-1:])
plt.figure(figsize=[30,30])
sns.heatmap(data.corr())
plt.show()

