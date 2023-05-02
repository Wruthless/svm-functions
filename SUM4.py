import pandas as pd
from matplotlib import pyplot
from matplotlib.pyplot import scatter
from pandas.plotting import scatter_matrix
from sklearn import svm, datasets

df = pd.read_csv(
    'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

# Size of the data, number of rows, number of columns.
print(df.shape)

# First ten rows.
print(f'\n', df.head(10))

# Last ten rows.
print(f'\n', df.tail(10))

# Summary of the data.
print(f'\n', df.describe())

# .sum() shows number of NaNs or missing numbers in each column.
# .sum().sum() shows the total number of NaNs or missing numbers of data.
print(f'\n', df.isna().sum().sum())

# Drop NaN elements.
df = df.dropna()
print(f'\n', df.groupby('species').size())

# Histogram.
df.hist()
# pyplot.show()

# Scatter plot matrix.
seplen_sep_width = df.loc[:, ['sepal_length', 'sepal_width']]
scatter_matrix(seplen_sep_width)
pyplot.show()

X = df.values[:, :2]
s = df['species']
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

# Predict
p = clf.predict([[5.4, 3.2]])
print(p)
