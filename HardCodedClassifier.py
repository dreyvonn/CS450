class HardCodedClassifier:
    def __init__(self, data_train, target_train, data_test):
        self.data_train = data_train
        self.target_train = target_train
        self.data_test = data_test

    def fit(self, data_train, target_train):
        model = target_train
        return model

    def predict(self, data_test):
        prediction = [0, 0, 0, 0, 0]
        return prediction