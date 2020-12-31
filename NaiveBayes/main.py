from sklearn.utils.validation import _num_samples
from NaiveBayes import NaiveBayes
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# X, y = make_classification(
#     n_samples=1000, n_features=10, n_classes=2, random_state=1234)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

# creating the model
clf = NaiveBayes()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy score
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
