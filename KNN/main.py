from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

knn_model = KNN(k=3)
knn_model.fit(X_train, y_train)
# predictions
y_pred = knn_model.predict(X_test)

# accuracy score
print(f"Accuracy score is: {accuracy_score(y_test, y_pred)}")
