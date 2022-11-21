## ModelZOO
from sklearn.linear_model import LogisticRegression
LogisticRegression_ = {
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__max_iter": [100, 150, 200, 250]
}

from sklearn.neural_network import MLPClassifier
MLPClassifier_ = {
    "mlpclassifier__alpha": [0.1, 1, 10, 100],
    "mlpclassifier__hidden_layer_sizes": [(32), (64), (64, 32)]
}

from sklearn.svm import SVC 
SVC_ = {
    "svc__C": [0.5, 0.6, 0.7, 1.0], 
    "svc__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    "svc__gamma": ["scale", "auto"]
}