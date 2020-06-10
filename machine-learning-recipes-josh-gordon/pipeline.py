# Description: This program classifies the iris
#              as three different species. 


# import a dataset 
from sklearn import datasets
iris = datasets.load_iris()

# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])

x = iris.data   #features
y = iris.target     #labels

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()


my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))





