from sklearn.model_selection import train_test_split
from sklearn import datasets
from LinearRegression import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# making regression data
X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1234)

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# Looking at the data
print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")

# creating the linear regression model
linreg = LinearRegression(lr=0.01)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

print(f"R2 score: {r2_score(y_test, y_pred)}")
print(f"Mean Squre Error: {mean_squared_error(y_test, y_pred)}")

# plotting the result
plt.figure()
plt.scatter(X_train, y_train, c='red')
plt.scatter(X_test, y_test, c='yellow')
plt.plot(X_test, y_pred)
plt.show()
