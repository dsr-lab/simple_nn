from abc import abstractmethod

class DatasetLoader:

    def __init__(self):
        self.num_classes = None
        self.classes = None
        self.train_ds = None
        self.test_ds = None

    def get_n_features(self):
        return next(iter(self.train_ds))[0].shape[0]

    @abstractmethod
    def init_dummy_features(self):
        pass


