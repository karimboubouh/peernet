class SklearnHook:
    def __init__(self, model):
        self.model = model
        self.score = 0

    def build(self):
        self.model = self.model()

    def train(self, *args, **kwargs):
        print(type(args))
        print(type(kwargs))
        # self.model.fit(*kwargs)
        self.help(*args, **kwargs)

    def evaluate(self, x_test, y_test):
        self.score = self.model.score(x_test, y_test)

    def help(self, x, y, **kwargs):
        print(type(kwargs))

    def summary(self):
        return {
            'test_test': self.score,
            'train_test': self.model,
        }


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
    model = SklearnHook(LogisticRegression)
    debug = 0
    model.train(x_train, y_train)
    # model.evaluate(x_test, y_test)
    # print(model.summary())
