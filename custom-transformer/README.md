# Machine-Learning-Tutorial
My self-learning about machine learning

# Customer transformer with Sklearn
Source [here](https://towardsdatascience.com/creating-custom-transformers-using-scikit-learn-5f9db7d7fdb5).

- Method 1:
`Transformers` are classes that enable data transformations while preprocessing the data for machine learning. Examples of transformers in Scikit-Learn are SimpleImputer, MinMaxScaler, OrdinalEncoder, PowerTransformer, to name a few. At times, we may require to perform data transformations that are not predefined in popular Python packages. In such cases, custom transformers come to the rescue.

- Method 2:
`FunctionTransformer` class of Scikit-Learn. This is a simpler approach that eliminates the need of defining a class, however, we need to define a function to perform the required transformation.