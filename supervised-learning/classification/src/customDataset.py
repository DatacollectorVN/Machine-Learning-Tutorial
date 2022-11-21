import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
import sys


class CustomDataset(object):
    def __init__(self, csv_file, output_col=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if output_col:
            self.outputs = self.data[output_col]
            self.inputs =  self.data.loc[:, self.data.columns != output_col]
        else:
            # get the last column as the ouput column
            self.outputs = self.data.iloc[:, self.data.shape[1] - 1]
            self.inputs =  self.data.iloc[:, :self.data.shape[1]]
            
        
    def __len__(self):
        return self.outputs.shape[0]
    
    def __getitem__(self, index):
        return self.inputs.iloc[index], self.outputs.iloc[index]

    def get_data(self, val_ratio, seed):
        X_train, X_val, y_train, y_val = train_test_split(self.inputs, self.outputs, test_size = val_ratio, random_state = seed)
        return X_train, X_val, y_train, y_val
    
    def get_orgi_data(self):
        return self.inputs, self.outputs


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None, special_transform=False, **args):
        self.drop_cols = drop_cols
        self.special_transform = special_transform
        for key in args:
            setattr(self, key, args[key])

        '''Edit Start'''
        # update new attribute or you can declare here
        # for example: 
        if self.special_transform:
            self.num_top_titles = 1
        '''Edit End'''
        
    def _drop_cols(self, X):
        return X.drop(self.drop_cols, axis=1)

    def fit(self, X, y=None):
        if self.special_transform:
            '''Edit Start'''
            # For example
            title_col = X.Name.str.extract(r"([a-zA-z]+)\.", expand=False)
            self.title_counts_ = title_col.value_counts()
            titles = list(self.title_counts_.index)
            self.top_titles_ = titles[: max(1, min(self.num_top_titles, len(titles)))]
            '''Edit End'''
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        if self.special_transform:
            '''Edit Start'''
            # For example
            title_col = X.Name.str.extract(r"([a-zA-z]+)\.", expand=False)
            title_col = title_col.transform(
                lambda x: x if x in self.top_titles_ else "Others"
            )
            X["Title"] = title_col
            '''Edit End'''
        
        if len(self.drop_cols) != 0:
            X = self._drop_cols(X)
            
        return X