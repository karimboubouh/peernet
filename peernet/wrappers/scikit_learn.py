from sklearn.metrics import log_loss, precision_score, recall_score, f1_score


class SklearnWrapper:
    def __init__(self, sk_model, **kwargs):
        self.sk_model = sk_model(**kwargs)
        self.metrics = {}

    def train(self, *args, **kwargs):
        self.sk_model.fit(*args, **kwargs)
        self.metrics['train_score'] = round(self.sk_model.score(*args, **kwargs), 4)
        try:
            self.metrics['train_size'] = args[0].shape[0]
        except (IndexError, AttributeError):
            pass

    def evaluate(self, *args, metrics=False, **kwargs):
        _score = self.sk_model.score(*args, **kwargs)
        self.metrics['test_score'] = round(_score, 4)
        if metrics:
            self._metrics(*args)
        return _score

    def _metrics(self, *args):
        pred = self.sk_model.predict(args[0])
        pred_proba = self.sk_model.predict_proba(args[0])
        self.metrics['test_loss'] = log_loss(args[1], pred_proba, eps=1e-15)
        self.metrics['precision'] = precision_score(args[1], pred, average="macro", zero_division=0)
        self.metrics['recall'] = recall_score(args[1], pred, average="macro", zero_division=0)
        self.metrics['f1_score'] = f1_score(args[1], pred, average="macro", zero_division=0)
        try:
            self.metrics['test_size'] = args[0].shape[0]
        except (IndexError, AttributeError):
            pass

    def info(self):
        return {
            'model': self.sk_model
        }

    def summary(self):
        return self.metrics

    @property
    def weights(self):
        return self.sk_model.coef_

    @weights.setter
    def weights(self, weights):
        assert (weights.shape == self.sk_model.coef_.shape)
        self.sk_model.coef_ = weights


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = SklearnWrapper(LogisticRegression)
    model.train(x_train, y_train)
    score = model.evaluate(x_test, y_test)
    print(score)
    print(model.summary()['test_score'])
