import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv('/Users/anupam7936/afb/train.csv', index_col='id')
X_test = pd.read_csv('/Users/anupam7936/afb/test.csv', index_col='id')

# Select target
y = data.target

# To keep things simple, we'll use only numerical predictors
X = data.drop(['target'], axis=1)
print(X.head())
# X = predictors.select_dtypes(exclude=['object'])

print('X.columns- {}'.format(X.columns))
print('X.columns data types- {}'.format(X.dtypes))

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train.columns if
                    X_train[cname].nunique() < 5 and
                    X_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if
                X_train[cname].dtype in ['int64', 'float64']]

# print('categorical_cols - {}'.format(categorical_cols))
# print('numerical_cols - {}'.format(numerical_cols))

def corr_col_to_be_dropped(df_train, corr_strength, col_cnt):
    '''
    :param df_train: Pandas dataframe having numerical features
    :param corr_strength: threshold to consider correlation strength between features
    :param col_cnt: number of columns to qualify the column to be dropped
    :return: list of columns to be dropped
    Usage example:
    col_drop = corr_col_to_be_dropped(X_train,0.41,1)
    '''
    cor_matrix = df_train.corr().abs()
    # print(type(cor_matrix))

    print(cor_matrix.shape)
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    print(upper_tri)

    to_drop = [column for column in upper_tri.columns if sum(upper_tri[column] > corr_strength) > col_cnt  ]
    print(to_drop)
    return to_drop

col_drop = corr_col_to_be_dropped(X_train,0.41,1)
print(col_drop)
X_train.drop(col_drop,axis=1)
print(X_train.columns)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
        # ,('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

preds_test = clf.predict(X_test)

# # Save test predictions to file
output = pd.DataFrame({'id': X_test.index,
                       'target': preds_test})
output.to_csv('/Users/anupam7936/afb/preds_test.csv', index=False)

