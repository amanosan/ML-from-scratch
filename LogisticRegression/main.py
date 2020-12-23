from sklearn.utils.validation import _num_samples
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# making our classification dataset
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

logreg = LogisticRegression(lr=0.0001, n_iters=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
