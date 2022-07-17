import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split

class CustomDataset(object):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, index_col = 0)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def get_data(self, val_ratio, seed):
        data_train, data_val = train_test_split(self.data, test_size = val_ratio, random_state = seed)
        return data_train, data_val
    
    def get_orgi_data(self):
        return self.data
    
class CustomTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, target_cols=None):
        self.target_cols = target_cols

    def _drop_col(self, X):
        cols = X.columns.to_list()
        self.drop_cols = [ele for ele in cols if ele not in self.target_cols]
        return X.drop(self.drop_cols, axis=1)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        return X
