# Description: This program classifies the fruits
#              as a apple or orange

from sklearn import tree

#features = [[140,"smooth"],[130,"smooth"],[150,"bumpy"],[170,"bumpy"]] #weight and bumpy
#labels = ["apple","apple","orange","orange"]

features = [[140, 1],[130, 1],[150, 0],[170, 0]]    #features
labels = [0, 0, 1, 1]   #labels

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))