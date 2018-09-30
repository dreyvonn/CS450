from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from HardCodedClassifier import HardCodedClassifier

iris = datasets.load_iris()
data = np.array(iris.data)
target = np.array(iris.target)
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, train_size=0.70,
                                                                    random_state=42)

classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

target_predicted = model.predict(data_test)

missed = 0
j = 0

for i in target_predicted:
    if i != target_test[j]:
        missed += 1
    j += 1

accuracy = int(((target_test.size - missed) / target_test.size) * 100)

print("Gaussian Accuracy = ", accuracy, "%\n")

classifier2 = HardCodedClassifier()
model2 = classifier2.fit(data_train, target_train)

target_predicted2 = model2.predict(data_test)

missed = 0

for i in range(len(target_predicted2)):
    if target_predicted2[i] != target_test[i]:
        missed += 1

accuracy2 = int(((target_test.size - missed) / target_test.size) * 100)

print("My Accuracy = ", accuracy2, "%\n")


