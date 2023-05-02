import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets
from matplotlib import pyplot
from matplotlib.pyplot import scatter
from pandas.plotting import scatter_matrix

# Example of SVM classification for breast cancer.
# 1. Load the breast cancer data.
# 2. Put data into a dataframe format.
# 3. Use 'features' data as X & 'target' data as y
# 4. Use train_test_split to split X and y into training and test sets.
# 5. 80% of data is for training, 20% for testing.
# 6. Use training data (X_train) and (y_train) to train an SVM.
# 7. Use test data X_test to make predictions.
# 8. Print the confusion matrix and classification report.

NUM_POINTS = 7
cancer = load_breast_cancer()

# All features.
X = cancer.data
df = pd.DataFrame(X, columns=[cancer.feature_names])
features_mean = list(df.columns[0:5])
features_name = cancer.feature_names[1:NUM_POINTS+1]

# All labels.
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

clf = SVC()
clf.fit(X_train, y_train)

# Scatter plot matrix.
fig = scatter_matrix(df[features_mean], figsize=((10,10)))
pyplot.show()

# Prediction
y_predict = clf.predict(X_test)

# Print confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['predicted_cancer', 'predicted_healthy'])

print(confusion)
print(classification_report(y_test, y_predict))
