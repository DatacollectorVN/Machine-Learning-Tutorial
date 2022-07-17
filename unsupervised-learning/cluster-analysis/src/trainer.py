from src.customDataset import CustomDataset, CustomTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from src.utils import find_optimal_k, str_to_class, convert_time, save_results
from sklearn.compose import make_column_transformer
import time

class SettingConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])


class Trainer(SettingConfig):
    def __init__(self, logger, **args):
        self.logger = logger
        super(Trainer, self).__init__(**args)
        self.K = range(self.K_RANGES[0], self.K_RANGES[1] + 1)
    
    def setup_data_loader(self):
        dataset = CustomDataset(self.DATA_DIR)
        return dataset.get_orgi_data()
    
    def create_pipeline(self, transformer=None, model_name=None):
        preprocessor = self._pipeline()
        model = str_to_class(model_name)
        if self.optimal_k:
            model = model(self.optimal_k, **self.MODELS_CONFIG[model_name])

        if transformer:
            pipeline = make_pipeline(transformer, preprocessor, model)  
        else:
            pipeline = make_pipeline(preprocessor, model)

        return pipeline

    def train(self):
        transformer = CustomTransformer(self.TARGET_COLS)
        data = self.setup_data_loader()
        preprocessor = self._pipeline()
        target_data = preprocessor.fit_transform(data)
        optimal_ks = []
        for model_name in self.MODELS:
            self.logger.log("ANNOUNCE", f"Using {model_name} model")
            if model_name == "KMeans":
                # Start finding optimal K 
                start_time = time.monotonic()
                self.optimal_k = find_optimal_k(self.K, target_data, self.MODELS_CONFIG[model_name])
                end_time = time.monotonic()
                mins, secs = convert_time(start_time, end_time)
                duration_find_k = f"{mins}m{secs}s"
                optimal_ks.append(self.optimal_k)

                # complete pipline with optimal K
                pipeline = self.create_pipeline(transformer = transformer, model_name = model_name)
                start_time = time.monotonic()
                pipeline.fit(data)
                end_time = time.monotonic()
                mins, secs = convert_time(start_time, end_time)
                duration_training_w_optimal_k = f"{mins}m{secs}s"
                save_results(pipeline, self.optimal_k, model_name, self.MODELS_CONFIG[model_name], 
                duration_find_k, duration_training_w_optimal_k)

            if model_name == "model2":
                pass
            
            self.logger.log("ANNOUNCE", f"Completed training {model_name} model")

    def _pipeline(self): 
        target_cols = self.TARGET_COLS
        target_transformer = Pipeline(steps = [
                ['target_minmax', MinMaxScaler()]
            ]
        )
        preprocessor = make_column_transformer(
            (target_transformer, target_cols)
        )
        return preprocessor
        