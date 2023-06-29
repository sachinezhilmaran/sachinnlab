1)a) Data Cleaning:

import pandas as pd
df=pd.read_csv("Data_set.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())
df.dropna(inplace='True')
x = df['height'].mean()
df['height'].fillna(x, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.drop_duplicates(inplace = True)


1)b) Detect and Remove Outliers:

import pandas as pd
import numpy as np
df=pd.read_csv("Data_set.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())
import matplotlib.pyplot as plt
plt.boxplot(data['price'])
plt.show()
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
upper_array = np.where(data['price']>=upper)[0]
lower_array = np.where(data['price']<=lower)[0]
data.drop(index=upper_array, inplace=True)
data.drop(index=lower_array, inplace=True)
plt.boxplot(data['price'])
plt.show()



2,3) Feature Selection Techniques:

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv("CarPrice.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())
df.dropna(inplace='True')
df.drop_duplicates(inplace = True)
data["fueltype"]=data["fueltype"].map({"gas":1,"diesel":0})
data.drop(['enginetype','carbody','symboling','CarName','aspiration','doornumber','drivewheel','enginelocation','cylindernumber','fuelsystem'], axis=1, inplace=True)
selector = SelectKBest(chi2, k=10)
data = selector.fit_transform(data, data["horsepower"]) #Enter the column that is of int64 datatype not float64
print("Selected Features:",data.shape)


4,5) Data Visualization:

import pandas as pd
data = pd.read_csv("CarPrice.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())
df.dropna(inplace='True')
df.drop_duplicates(inplace = True)
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data['horsepower'])
plt.show()
sns.countplot(data['horsepower'])
plt.show()
sns.histplot(data['horsepower'])
plt.show()
sns.lineplot(data['horsepower'])
plt.show()
x = data['fueltype'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.show()
sns.barplot(x=data['fueltype'],y=data['horsepower'])
plt.show()
sns.scatterplot(data['horsepower'])
plt.show()
sns.heatmap(data['horsepower'].corr(), annot=True)
plt.show()
