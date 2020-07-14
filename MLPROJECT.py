#---------------------Phase-I=>data cleaning------------------------------------------------
import pandas as pd

df = pd.read_csv("C:\\Users\\nick\\RetailDatalog")
print(df)

#------------------ProductCategory,Outlet_Type --> junk--------------------------------------

import re
df['ProductCategory'] = [re.sub(r'\W', '', i) for i in df['ProductCategory']]
df['Outlet_Type'] = [re.sub(r'\W', '', i) for i in df['Outlet_Type']]
print(df)

#-------------------null values.... (replace/mean, sum())------------------------------------

df.isnull().sum()

#-------------------null replace by zero-----------------------------------------------------
df['ProductWeight'] = df['ProductWeight'].fillna(0, inplace=False) #filling missing values
df['Outlet_Size'] = df['Outlet_Size'].fillna(0, inplace=False) #filling missing values
df.Outlet_Size = df.Outlet_Size.astype(str) #resolving type error

#--------------convert text to numerous------------------------------------------------------
from sklearn import preprocessing
l_encoder = preprocessing.LabelEncoder()   #New Object
df = df.apply(l_encoder.fit_transform)
print(df)
#----------missing values replace by mean()--------------------------------------------------
df['Outlet_Size'] = df['Outlet_Size'].replace(0, df['Outlet_Size'].mean()) #replacing zero values with df['Outlet_Size']
df.Outlet_Size = round(df.Outlet_Size, 2) #rounding off
#print(df)

#------------------------------------Visualisation-- seaborn---------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot = True, fmt = '.3f')
plt.show()
#Data Cleaning ends here.
#------------------------------export to csv(Final Data) ------------------------------------

df.to_csv("C:\\Users\\nick\\RetailDatalog.csv")
#df.drop('ProductID', axis=1)
#df.drop('Outlet-ID', axis=1)
#--------------------------------------------------------------------------------------------------
#==========================LINEAR REGRESSION-->DECIDED BY CALCULATED VALUES===========================
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
df = pd.read_csv("C:\\Users\\nick\\RetailDatalog.csv")
#mapping input and target variable
#X = df['ht']
#Y = df['wt']

X = df.drop('Item_Outlet_Sales', axis=1)

Y = df['Item_Outlet_Sales']
#===================data splitting into two parts (train & test)=====================================
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5)
#test_size always < 0.5
print(x_train)
print(x_test)
print(y_train)
print(y_test)
#===================3rd step-> Linear Regression======================================================
from sklearn.linear_model import LinearRegression
model = LinearRegression()
modelfit = model.fit(x_train, y_train) #train the model
ycap = model.predict(x_test)
print(ycap)

#df['ycapnew'] = df['yap']>0.5,1,0
ycapnew = []
for x in ycap :
    if x < 0.5 :
        x = 0
    else:
        x = 1
    ycapnew.append(x)
dfcap = pd.DataFrame(ycapnew)
print(dfcap)
print(y_test)

#=================================accuracy================================================
from sklearn.metrics import accuracy_score
print('accuracy:-',accuracy_score(y_test, ycapnew))

#================K-fold cross validation-> to improve accuracy score============================
from sklearn.model_selection import  cross_val_score,cross_val_predict
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
crossycap = cross_val_predict(modelfit, df, Y, cv=4 )
score = r2_score(Y, crossycap.round())
print('k-fold score:-',score)

#=====================================rmse====================================================
def rmse(Y, ycap):
    score = np.sqroot(np.mean(((Y-ycap.round())**2)))
    return score
print('rmse score:', score)

#=============================clustering part (recommendation engine)========================
import matplotlib.pyplot as plt
plt.scatter(df["Outlet_Location_Type"], df["ProductCategory"])
plt.show()

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, init="k-means++")
prediction = model.fit_predict(df[["Outlet_Location_Type", "ProductCategory"]])
df["cluster"] = prediction
print(df)
df.to_csv("C:\\Users\\nick\\RetailDatalog1.csv")
#=============================================================================================