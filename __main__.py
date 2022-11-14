from sklearn.model_selection import train_test_split
from sklearn import datasets
from knn import *

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

'''
    Here we check accuracy for different k to compare them between each other.
    Such as we have odd amount of labels, we can use even amount of k, other way if
    we have even amount of labels we should use odd amount of labels
    # k_range = range(1, 10, 2), with step - 2
'''
k_range = range(1, 10)
scores = []
for k in k_range:
    knn = KNN(n_neighbors=k)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test)
    scores.append(accuracy(y_test, predict))

knn2 = KNN(n_neighbors=3)
knn2.fit(X_train, y_train)
predicted = knn2.predict(X_test)
# represent(predicted)
print("Accuracy of Classification is: ", accuracy(y_test, predicted))
#%%
