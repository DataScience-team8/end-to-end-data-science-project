# end-to-end-data-science-project
# Advantage
> Predict Wine Quality
> Preparing products based on attribute values that are highly correlated with quality can be beneficial
> The model allows for quick qualification after ingredient analysis, which can have a positive impact on pricing time.
> It's now possible to measure quality even when ingredients are scarce
# Table
- Data
- Preprocessing
- Evaluate
- Regression
1. Linear Regression
- Classification
1. Rogistic Regression
2. Decision Tree
3. Knn
4. RandomForest
- Special Trial
  1. Feature Engineering

# Data
- feture: fixed acidity,	volatile acidity,	citric acid,	residual sugar,	chlorides,	free sulfur dioxide,	total sulfur dioxide,	density,	pH,	sulphates,	alcohol
- target: quality
### Data Preprocessing
- Standard Nomalization
- feture Engineering: restore feature with linear regression, use polynomial regression
- Data Split: StratifiedKFold
### Data Analizing
- target: 1~10 integer, so we can use regression and clustering
# Model Evaluation
### Regression
- Linear Regression with Feature Engineering
### Clustering
- Logistic Regression
- Decision Tree
- KNN
- RandomForest
# Special trial
### Regression Model
- Polinomial Regression

