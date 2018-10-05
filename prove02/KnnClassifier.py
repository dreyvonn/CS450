from KnnModel import KnnModel


class KnnClassifier:

    def __init__(self, k):
        self.k = k

    def fit(self, data, target):
        model = KnnModel(data=data, target=target, k=self.k)
        return model
