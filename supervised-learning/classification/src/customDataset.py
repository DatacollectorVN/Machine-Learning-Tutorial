import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
    

class CustomDataset(object):
    def __init__(self, csv_file, output_col=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if output_col is None:
            # get the last column as the ouput column
            self.outputs = self.data.iloc[:, self.data.shape[1] - 1]
            self.inputs =  self.data.iloc[:, :self.data.shape[1]]
        else:
            self.outputs = self.data[output_col]
            self.inputs =  self.data.loc[:, self.data.columns != output_col]
        
    def __len__(self):
        return self.outputs.shape[0]
    
    def __getitem__(self, index):
        return self.inputs.iloc[index], self.outputs.iloc[index]

    def get_data(self, val_ratio, seed):
        X_train, X_val, y_train, y_val = train_test_split(self.inputs, self.outputs, test_size = val_ratio, random_state = seed)
        return X_train, X_val, y_train, y_val
    
    def get_orgi_data(self):
        return self.inputs, self.outputs


class CustomTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, num_top_titles=1, drop_cols=None):
        self.drop_cols = drop_cols
        self.num_top_titles = num_top_titles
        
    def _drop_cols(self, X):
        return X.drop(self.drop_cols, axis=1)

    def fit(self, X, y=None):
        title_col = X.Name.str.extract(r"([a-zA-z]+)\.", expand=False)
        self.title_counts_ = title_col.value_counts()
        titles = list(self.title_counts_.index)
        self.top_titles_ = titles[: max(1, min(self.num_top_titles, len(titles)))]
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        title_col = X.Name.str.extract(r"([a-zA-z]+)\.", expand=False)
        title_col = title_col.transform(
            lambda x: x if x in self.top_titles_ else "Others"
        )
        X["Title"] = title_col
        
        X = self._drop_cols(X)
        return X