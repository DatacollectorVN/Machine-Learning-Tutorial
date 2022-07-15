# Machine-Learning-Tutorial
My self-learning about machine learning

## Supervised-learning - Classification models

Some common Supervised-learning models that is supported by `Sklearn`:
- Logistics Regression
- Support Vector Machine (SVM)
- KNN
- Decision Tree
- Random Forest
- Linear Discriminant Analysis (LDA)
- Artificial Neural Network (ANN)

## ML-Classification project architecture

### Setup 
- Create virtual environment
```bash
conda create -n ml-tutorial python=3.8 -y
conda activate ml-tutorial
```
- Install prerequirements
```bash
pip install -r requirements.txt
```

### For training
- Change some configuration, change content of file `config/train.yaml`.
- Start training
```bash
python train.py
```

### Add more ML models
This repository use `Grid Search` technique for finding best parameter for each model. Therefore, the parametets for tunning each model in `src/models.py`. Please read that carefully and add some approriate parameter's names in `GridSearchCV`. Please read our tuturial first it you'r not familiar with it in `custom-transformer/class-inheritance`.

- Syntax for adding new model
```bash
from sklearn.... import <MODEL_NAME>
<MODEL_NAME>_ = {
    <PARAMETER_NAME_1>: <LIST_OF_PARAMTER_VALUE>,
    <PARAMETER_NAME_2>: <LIST_OF_PARAMTER_VALUE>, 
    ...
}

```
- Example of parameter's name of `MLPClassifier`:
```bash
MLPClassifier_ = {
    "mlpclassifier__alpha": [0.1, 1, 10, 100],
    "mlpclassifier__hidden_layer_sizes": [(32), (64), (64, 32)]
}
```

