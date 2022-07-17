from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from math import sqrt
from datetime import datetime
import os
import joblib
import numpy as np
import json
from loguru import logger
import sys

def find_optimal_k(K, target_data, config_k_mean, save_figure=False):
    distortions = _calculate_distortions(K, target_data, config_k_mean)
    optimal_k = _find_optimal_k(K, distortions)
    if save_figure:
        pass
    return optimal_k

def _calculate_distortions(K, target_data, config_k_mean):
    distortions = []
    for k in K:
        kmeanmodel = KMeans(n_clusters = k, **config_k_mean) 
        kmeanmodel.fit(target_data)
        distortions.append(sum(np.min(cdist(target_data,
                                            kmeanmodel.cluster_centers_, "euclidean"), axis = 1)) \
                                            / target_data.shape[0]
        )
    return distortions

def _calc_distance(x1, y1, a, b, c):
  d = abs((a * x1 + b * y1 + c)) / (sqrt(a * a + b * b))
  return d

def _find_optimal_k(K, distortions):
    a = distortions[0] - distortions[-1]
    b = K[-1] - K[0]
    c1 = K[0] * distortions[-1]
    c2 = K[-1] * distortions[0]
    c = c1 - c2
    distance_of_points_from_line = []
    for k in range(len(K)):
        distance_of_points_from_line.append(
            _calc_distance(K[k], distortions[k], a, b, c)
        )
    optimal_k = K[distance_of_points_from_line.index(max(distance_of_points_from_line))]
    return optimal_k
    
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def convert_time(start_time, end_time):
    '''
    Convert time (miliseconds) to minutes and seconds
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_results(pipeline, optimal_k, model_type, model_config, 
    duration_find_k, duration_training_w_optimal_k):
    os.makedirs("experiments", exist_ok = True)
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    base_dir = os.path.join("experiments", now)
    os.mkdir(base_dir)
    _save_meta_experiment(optimal_k, model_type, base_dir, model_config,
    duration_find_k, duration_training_w_optimal_k)
    _save_pipeline(pipeline, base_dir, model_type)

def _save_meta_experiment(optimal_k, model_type, base_dir, model_config,
    duration_find_k, duration_training_w_optimal_k):
    dct = {}
    dct['optimal_k'] = optimal_k
    dct['model_type'] = model_type
    dct['model_config'] = model_config
    dct['duration_find_k'] = duration_find_k
    dct['duration_training_w_optimal_k'] = duration_training_w_optimal_k
    with open(os.path.join(base_dir, "meta_experiment.json"), "w") as outfile:
        json.dump(dct, outfile)

def _save_pipeline(pipeline, base_dir, model_type):
    joblib.dump(pipeline, os.path.join(base_dir, f"best_{model_type}.pkl"), compress = 1)

def custom_loggers():
    logger.level("BUG", no=38, color="<red>")
    logger.level("ANNOUNCE", no=38, color="<yellow>")
    return logger