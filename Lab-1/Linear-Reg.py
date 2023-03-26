import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import Lasso,Ridge


df=pd.read_csv('/content/company.csv')
d=df[["TV","Sales"]]

xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state=2)


model = linear_model.LinearRegression()
model.fit(xtr,ytr)
ypre = model.predict(xte)
MSEloss = mean_squared_error(yte,ypre)
MAEloss = mean_absolute_error(yte,ypre)
f = r2_score(yte,ypre)
print(MSEloss,MAEloss,f)


#Lasso Reg
r = Lasso(alpha=15.0)
r.fit(xtr,ytr)

y_p = model.predict(xte)
nmsl = mean_squared_error(yte,y_p)
MSEloss = mean_squared_error(yte,y_p)
MAEloss = mean_absolute_error(yte,y_p)
f = r2_score(yte,y_p)
print(nmsl,MAEloss,f)
