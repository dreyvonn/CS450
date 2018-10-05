import numpy as np
from sklearn.preprocessing import StandardScaler


class KnnModel:
    def __init__(self, data, target, k):
        self.data = data
        self.target = target
        self.k = k

    def predict(self, data_test):

        # Scale the data
        scaler = StandardScaler()
        scaler.fit(self.data)
        data_new = scaler.transform(self.data)

        scaler.fit(data_test)
        test_new = scaler.transform(data_test)

        # Create array to place predictions
        prediction_arr = np.array([], dtype=int)

        # Loop through the test data and getting a prediction for each one
        for i in range(len(test_new)):
            single_prediction = predict_single(test_new[i], data_new, self.k, self.target)
            prediction_arr = np.append(prediction_arr, single_prediction)

        return prediction_arr


def predict_single(data_single, data, k, target):

    # Array to place the distance for each set in the training data
    distance_arr = np.array([])

    # Calculate the distance between the test set and each training set
    for i in range(len(data)):
        distance = (data_single[0] - data[i][0]) ** 2 + (data_single[1] - data[i][1]) ** 2 + \
                   (data_single[2] - data[i][2]) ** 2 + (data_single[3] - data[i][3]) ** 2
        distance_arr = np.append(distance_arr, distance)

    # Find the indices of k nearest neighbors
    index = distance_arr.argsort()[:k]

    # Get the target value of the nearest neighbors and place them into an array
    predictions = np.array([], dtype=int)
    for i in range(len(index)):
        predictions = np.append(predictions, target[index[i]])

    # Returns the most common target value in the array
    counts = np.bincount(predictions)
    single_prediction = np.argmax(counts)

    return single_prediction
