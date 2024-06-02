# end-to-end-data-science-project
# Table
- Data
- Preprocessing
- Evaluate
- Regression
1. Linear Regression
2. Rogistic Regression
- Clustering
1. Decision Tree
2. Knn
3. RandomForest
- Special Trial
  1. Feature Engineering

# Data
- feture: fixed acidity,	volatile acidity,	citric acid,	residual sugar,	chlorides,	free sulfur dioxide,	total sulfur dioxide,	density,	pH,	sulphates,	alcohol
- target: quality
### Data Preprocessing
- Standard Nomalization
- feture Engineering: restore feature with linear regression, use polynomial regression
- Data Split: StratifiedKFold
```
# Separate white wine data
white = data[data['type']=='white']
white = white.drop('type', axis=1)
white = white.reset_index()
print(white.isna().sum())

# Use columns without any missing values to predict missing values
non_nan_columns = white.loc[:, white.isna().sum() == 0].columns  # Columns without any missing values
nan_columns = white.loc[:, white.isna().sum() != 0].columns # Columns with missing values

drop_white = white.dropna()  # Clean data without missing values
drop_white_X = drop_white[non_nan_columns]
for col in nan_columns: # For each column with missing values
    drop_white_y = drop_white[col] # Set the column as the target variable
    model = LinearRegression()
    model.fit(drop_white_X, drop_white_y)

    nan_idx = white.loc[white[col].isnull(), col].index
    print(nan_idx)
    white.loc[white[col].isnull(), col]= model.predict(white[non_nan_columns].iloc[nan_idx, :])# Fill missing values

print(white.isna().sum())

white_X = white.drop('quality', axis=1)
white_y = white['quality']

# Scale the features
for col in white_X.columns:
    white_X[col] = StandardScaler().fit_transform(white_X[[col]])

# Split the data into training and testing sets
white_X_train, white_X_test, white_y_train, white_y_test = train_test_split(white_X,white_y,
                                                    test_size = 0.2,
                                                    random_state=1,
                                                    stratify=white_y) # Maintain class distribution

```
### Data Analizing
- target: 1~10 integer, so we can use regression and clustering
# Model Evaluation
### Regression
- Linear Regression with Feature Engineering
- Rogistic Regression
### Clustering
- Dicision Tree
- KNN
- RandomForest
# Special trial
### Regression Model
- Polinomial Regression

