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
### For running with Docker
It seems like pretty simple.
```bash
docker-compose up -d
``` 
*Expected output:* Create `experiments` folder to record all training experiments.
### For running without Docker 
- Create virtual environmentv
```bash
conda create -n ml-tutorial python=3.8 -y
conda activate ml-tutorial
```
- Install prerequirements
```bash
pip install -r requirements.txt
```
- Setup dataset
```bash
bash setup_data.sh
```
### For training
- Change some configuration, change content of file `config/train.yaml`.
- Start training
```bash
python train.py
```

## Add more ML models
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

## Custom training for your own dataset
### Train configuration
All your configuration in `config/train.yaml`, `src/models` and `src/trainer` with:
- `config/train.yaml`: Basic control the data processing / transformation and training experiment.
- `src/models`: List the model that have potential to be used in the training experiment and their hyperparametes for tunning by using **GridSearchCV**.
- `src/trainer`: Allow you can custom the `CustomTransformer` class. *Note: You can edit at the block inside `Edit Start / End`*. 

Please following our intrusctions:
#### With `config/train.yaml`
- **MODELS: (List[Str])** List of models name that are used in experiemnt. Please provide correct name that corresponding class name from sklearn that you import. 

I.e: the Logistic Regression model have name from sklearn is `from sklearn.linear_model import LogisticRegression` --> so name is `LogisticRegression`.

- **DATA_DIR: (Str)** The data directory of dataset. *Note: currently, just allow `csv` file*.

- **SPECIAL_TRANSFORM: (Boolean)** If you want to custom the `CustomTransformer` from `src/trainer` for specific tasks choose `True`. If don't choose `False`.

- **TARGET_COL: (Str)** The name of columns is ground truth of dataset. If you don't give any value this automatic choose the last column of table.

- **DROP_COLS: (List[Str])** The list name of columns that will be dropped. If don't drop any columns --> empty list (`[]`)

- **CONTINUOUS_COLS: (List[Str])** The list name of continuous columns. If don't drop any columns --> empty list (`[]`)

- **UNORDER_CATE_COLS: (List[Str])** The list name of unorder categorical columns. If don't drop any columns --> empty list (`[]`)

- **ORDER_CATE_COLS: (List[Str])** The list name of order categorical columns. If don't drop any columns --> empty list (`[]`)

- **PARAM_GRID: (Dict)** Contain all the parameters to tunning for data transformation. 
*Note:* It do not contain the parameters of models and if you don't provide any column for `CONTINUOUS_COLS`, `UNORDER_CATE_COLS` and `ORDER_CATE_COLS`. Then just give it empty dictory `PARAM_GRID: {}`

- **GRID_SEARCH_CV_CONFIG: (Dict)** Contain all the parameter of `GridSearchCV`, read [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for understand more.