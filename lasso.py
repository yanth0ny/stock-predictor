import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime

from sklearn.linear_model import Lasso
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

start = datetime.datetime(2002, 5, 23)
end = datetime.datetime(2019, 8, 30)
df = web.DataReader("NFLX", 'yahoo', start, end)


forecast_out = int(200) # predicting 30 days into future
df['Prediction'] = df[['Adj Close']].shift(-forecast_out) #  label column with data shifted 30 units up

x = np.array(df.drop(['Prediction'], 1))
x = preprocessing.scale(x)

x_forecast = x[-forecast_out:] # set x_forecast equal to last 30
x = x[:-forecast_out] # remove last 30 from x
y = np.array(df['Prediction'])
y = y[:-forecast_out]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Training
clf = Lasso()
clf.fit(x_train,y_train)
# Testing
confidence = clf.score(x_test, y_test)
forecast_prediction = clf.predict(x_forecast)

last_date = df.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_prediction:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj Close'].tail(1000).plot()
df['Prediction'].tail(200).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
