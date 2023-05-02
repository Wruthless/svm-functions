from sklearn import svm, datasets

iris = datasets.load_iris()

# Sepal length, Sepal width, Petal length, Petal width
X = iris.data[:, :4]

# 0: Setosa, 1: Versicolour, 2: Virginica
y = iris.target
print(y)

clf = svm.SVC()
clf.fit(X, y)

# Prediction
# Predict the flower for a given Sepal and Petal length/width
p = clf.predict([[5.4, 3.2, 4.4, 2.2]])
print(p)
