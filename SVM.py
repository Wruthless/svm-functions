from sklearn import svm

# Height[cm], Weight[kg], Shoesize[UK
X = [
    [170, 70, 10],
    [180, 80, 12],
    [170, 65, 8],
    [160, 55, 7],
    [155, 68, 9],
    [175, 62, 11]
]

# Gender 0: Male, 1: Female
y = [0, 0, 1, 1, 1, 0]

clf = svm.SVC()
clf.fit(X, y)

# Prediction
p = clf.predict([[160, 60, 7]])
print(p)
