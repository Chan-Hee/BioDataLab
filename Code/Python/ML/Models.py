class Models(object):
    def __init__(self):
        pass

    def get_name(self):
        raise NotImplementedError()

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def test(self , x, y):
        raise NotImplementedError()

