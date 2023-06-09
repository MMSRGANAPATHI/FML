# -*- coding: utf-8 -*-
"""startups(ridge)-Lr.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jGabXUAD5QuiKMqk20P2fnySrvQwr7ZM
"""

import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import  mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import Ridge

df = pd.read_csv('/50_Startups.csv')
df

from google.colab import drive
drive.mount('/content/drive')

df.info

df.isnull().sum()

type(df['State'])

x = df.drop(['State','Profit'],axis=1)
y = df['Profit']
x,y

xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state=42)

xtr.shape,xte.shape

reg = linear_model.LinearRegression()
reg.fit(xtr,ytr)

ypre = reg.predict(xte)
mael = mean_absolute_error(yte,ypre)
msel = mean_squared_error(yte,ypre)
rs = r2_score(yte,ypre)
print(mael)
print(msel)
print(rs)

reg = linear_model.Ridge(alpha = 0.5)
reg.fit(xtr,ytr)

ypre1 = reg.predict(xte)
mael = mean_absolute_error(yte,ypre)
msel = mean_squared_error(yte,ypre)
rs = r2_score(yte,ypre)
print(mael)
print(msel)
print(rs)