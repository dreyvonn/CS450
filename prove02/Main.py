from KnnClassifier import KnnClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
data = np.array(iris.data)
target = np.array(iris.target)
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, train_size=0.70,
                                                                    random_state=40)
# ............................................................................
# Sk-learn KNN classifier
# ............................................................................
classifier_Sk = KNeighborsClassifier(n_neighbors=3)
model_Sk = classifier_Sk.fit(data_train, target_train)
prediction_Sk = model_Sk.predict(data_test)

missed_Sk = 0
for i in range(len(prediction_Sk)):
    if prediction_Sk[i] != target_test[i]:
        missed_Sk += 1

accuracy_Sk = int(((prediction_Sk.size - missed_Sk) / prediction_Sk.size) * 100)

print("Sk-learn prediction = \n", prediction_Sk)
print("Sk-learn accuracy = ", accuracy_Sk, "%")

# ............................................................................
# My KNN classifier
# ............................................................................
classifier = KnnClassifier(k=4)
model = classifier.fit(data_train, target_train)
prediction = model.predict(data_test)

missed = 0
for i in range(len(prediction)):
    if prediction[i] != target_test[i]:
        missed += 1

accuracy = int(((prediction.size - missed) / prediction.size) * 100)

print("\nMy prediction = \n", prediction)
print("My accuracy = ", accuracy, "%")
