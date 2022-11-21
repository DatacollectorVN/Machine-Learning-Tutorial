from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from src.customDataset import CustomDataset, CustomTransformer
from src.utils import save_results, str_to_class, convert_time
import time
import sys

class SettingConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])


class Trainer(SettingConfig):
    def __init__(self, logger, **args):
        self.logger = logger
        super(Trainer, self).__init__(**args)

    def setup_data_loader(self):
        dataset = CustomDataset(self.DATA_DIR, self.TARGET_COL)
        return dataset.get_orgi_data()
    
    def create_pipeline(self, transformer, model_name):
        preprocessor = self._pipeline()
        model = str_to_class(model_name)
        if preprocessor:
            pipeline = make_pipeline(transformer, preprocessor, StandardScaler(), model())
        else:
            pipeline = make_pipeline(transformer, StandardScaler(), model())
    
        return pipeline
    
    def train(self):
        transformer = CustomTransformer(self.DROP_COLS, self.SPECIAL_TRANSFORM)
        X, y = self.setup_data_loader()
        for model_name in self.MODELS:
            self.logger.log("ANNOUNCE", f"Using {model_name} model")
            pipeline = self.create_pipeline(transformer, model_name)
            param = str_to_class(f"{model_name}_")
            param_grid = self.PARAM_GRID.copy()
            
            param_grid.update(param)
            print(param_grid)
            # sys.exit()
            estimator = GridSearchCV(
                pipeline, 
                param_grid, 
                **self.GRID_SEARCH_CV_CONFIG
            )
            start_time = time.monotonic()
            estimator.fit(X, y)
            
            end_time = time.monotonic()
            mins, secs = convert_time(start_time, end_time)
            duration = f"{mins}m{secs}s"
            save_results(estimator, model_name, duration)
            self.logger.log("ANNOUNCE", f"Completed training {model_name} model")

    def _pipeline(self):
        continuous_cols = self.CONTINUOUS_COLS
        unorder_cate_cols = self.UNORDER_CATE_COLS
        order_cate_cols = self.ORDER_CATE_COLS
        column_transformers = []

        if len(continuous_cols) != 0 or continuous_cols is None:
            numeric_transformer = Pipeline(steps = [
                    # num_imputer tunning ['Mean', 'Median']
                    ['num_imputer', SimpleImputer()]
                ]
            )
            column_transformers.append((numeric_transformer, continuous_cols))

        if len(unorder_cate_cols) != 0 and unorder_cate_cols is not None:
            unorder_categorical_transformer = Pipeline(steps = [
                    ['unorder_imputer', SimpleImputer(strategy = "most_frequent")],
                    ['unorder_onehot',OneHotEncoder(handle_unknown = "ignore")]
                ]
            )
            column_transformers.append((unorder_categorical_transformer, unorder_cate_cols))

        if len(order_cate_cols) != 0 and order_cate_cols is not None:
            order_categorical_transformer = Pipeline(steps = [
                    ['order_imputer', SimpleImputer(strategy = "most_frequent")]
                ]
            )
            column_transformers.append((order_categorical_transformer, order_cate_cols))
        
        preprocessor = None
        if len(column_transformers) != 0:
            preprocessor = make_column_transformer(
                *column_transformers,
            )
        
        return preprocessor
    
    

    
        