import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# generate random numbers
np.random.seed(0)

# rp.random.randn(n) gives sample og n float numbers with mean 0 and stdv 1
X = 2.5 * np.random.randn(100) + 1.5 # mean is now 1.5 and stdv 2.5
e = 0.5* np.random.randn(100) # error mistakes from y^ - y

y = 2 + 0.7*X + e # actual value of y

# pandas dataframe to store values of X and y
df = pd.DataFrame(
    {'X': X,
     'y': y}
)

# print(df.head())

# need mean to estimate b0 and b1 or a and b
xmean = np.mean(X)
ymean = np.mean(y)

# can take sum((x - x_)*(y - Y_))
df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['X'] - xmean)**2

beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)

print("alpha", alpha)
print("beta", beta)

ypred = alpha + beta*X

print("y^ = ", ypred)

plt.figure(figsize=(12, 6))
plt.plot(X, ypred)
plt.plot(X, y, 'ro')
plt.title('Actual vs predicted')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()


# Solving the problem vwith use of ols method in statsmodel
import statsmodels.formula.api as smf

advert = pd.read_csv("Advertising.csv")
print(advert.head())

# intialise and fit linear regression model using statsmodel
model = smf.ols('sales ~ TV', data=advert)
model = model.fit()

# values of alpha b_0 intercept, and b_1 beta
print(model.params)

# predict values
sales_pred = model.predict()

#plot

plt.figure(figsize=(12, 6))
plt.plot(advert['TV'], sales_pred, 'r', linewidth=2)
plt.plot(advert['TV'], advert['sales'], 'o')
plt.title('TV advertising cost')
plt.xlabel('Sales')
plt.ylabel('TV vs Sales')

plt.show()

# solving with sklearn
from sklearn.linear_model import LinearRegression

# solving for mutilpe regression
predictors = ['TV', 'radio']

X = advert[predictors]
y = advert['sales']

# initilize and fir model
lm = LinearRegression()
model = lm.fit(X, y)

# values of alpha b_0 intercept, and b_1 beta
print(model.intercept_)
print(model.coef_)

sales_pred = model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(advert[predictors], sales_pred, 'r', linewidth=1)
plt.plot(advert[predictors], advert['sales'], 'o')
plt.title('TV advertising cost')
plt.xlabel('Sales')
plt.ylabel('TV vs Sales')

plt.show()
