import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("tvmarketing.csv")

df.head()

df.tail()

df.info()

df.shape

df.describe()

d=df.plot(x ='TV', y='Sales', kind = 'scatter')

X=df['TV']
X.head()

y=df['Sales']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))

print((X_train.shape))
print((X_test.shape))
print((y_train.shape))
print((y_test.shape))

X_train=X_train[:, np.newaxis]
X_test=X_test[:, np.newaxis]

print((X_train.shape))
print((y_train.shape))
print((X_test.shape))
print((y_test.shape))

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print('Coefficients: ', regr.coef_)
print('intercept: ',regr.intercept_)

from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error
print('Mean squared error: %.2f'% mean_squared_error(y_test,y_pred))

# The coefficient of determination: 1 is perfect prediction
print(' r^2 Coefficient of determination: %.2f'% r2_score(y_test,y_pred))

# Plot outputs
plt.scatter(X_test,y_test,  color='green')
plt.plot(X_test,y_pred, color='blue', linewidth=1)
plt.xlabel("TV")
plt.ylabel("Sales")

#plt.xticks(())
#plt.yticks(())

plt.show()

