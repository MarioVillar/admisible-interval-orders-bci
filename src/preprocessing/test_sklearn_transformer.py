from sklearn.base import TransformerMixin


class PrintTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.shape)
        return X
