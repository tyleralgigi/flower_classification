import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump, load
from sklearn import preprocessing
import random

# load the data
cols = ['sepal_length', ' sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=cols)

#generate a pair plot for the Iris dataset
sns.pairplot(iris, hue='class', size=2.5);
############

#Starter code
plt.show()

iris['class'].unique()
iris['class_encod'] = iris['class'].apply(lambda x: 0 if x == 'Iris-setosa' else 1 if x == 'Iris-versicolor' else 2)
iris['class_encod'].unique()
y = iris[['class_encod']] # target attributes 
X = iris.iloc[:, 0:4] # input attributes
X.head()
y.head()
############

#normalize the features of the iris dataset
labels = iris[['class']] 
x = iris[['sepal_length', ' sepal_width', 'petal_length', 'petal_width']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
iris=pd.DataFrame(x_scaled, columns=df.columns)   
iris['class'] = labels
############

#train the model (Starter code)
random.seed(42) # for reproducibility purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)

np.shape(y_train)

m = KNeighborsClassifier()
m.fit(X_train, np.ravel(y_train))
m.predict(X_test.iloc[0:10])
y_test[0:10]
m.score(X_test, y_test)

# Plot non-normalized confusion matrix (Starter code)
from sklearn.metrics import plot_confusion_matrix

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(m, X_test, y_test,
                                 display_labels=iris['class'].unique(),
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

plt.show()

dump(m, 'iris-classifier.dmp')

ic = load('iris-classifier.dmp')
confusion_matrix(y_test, ic.predict(X_test))
############


#build a decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree,metrics 

dtc = DecisionTreeClassifier()
clf = dtc.fit(X_train, np.ravel(y_train))
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=iris.columns,  
                   class_names=iris['class'].unique(),
                   filled=True)

fig.savefig("decistion_tree.png")
############