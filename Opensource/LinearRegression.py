from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error,mean_absolute_percentage_error

# Select features with correlation values greater than |0.2|
new_X_train = pd.DataFrame({'alcohol':white_X_train['alcohol'],
                            'density':white_X_train['density'],
                            'volatile acidity':white_X_train['volatile acidity'],
                            'chlorides':white_X_train['chlorides']})
new_X_test = pd.DataFrame({'alcohol':white_X_test['alcohol'],
                            'density':white_X_test['density'],
                            'volatile acidity':white_X_test['volatile acidity'],
                            'chlorides':white_X_test['chlorides']})
# Train the Linear Regression model
model = LinearRegression()
model.fit(new_X_train, white_y_train)
# Predict using the model
y_predict = model.predict(new_X_test)

# evaluation
print("r2 : {}".format(r2_score(white_y_test, y_predict)))
print("MAE : {}".format(mean_absolute_error(white_y_test, y_predict)))
print("MSE : {}".format(mean_squared_error(white_y_test, y_predict)))
print("MAPE : {}".format(mean_absolute_percentage_error(white_y_test, y_predict)))

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

poly = PolynomialFeatures(degree=2,include_bias=False)

# white wine
new_X_train_white = white_X_train
new_X_test_white = white_X_test

poly_X_train_white = poly.fit_transform(new_X_train_white)
poly_X_test_white = poly.transform(new_X_test_white)

model_poly = LinearRegression()
model_poly.fit(poly_X_train_white, white_y_train)
y_predict_poly_white = model_poly.predict(poly_X_test_white)


print("White Wine - Polynomial Regression")
print("r2 : {:.2f}".format(r2_score(white_y_test, y_predict_poly_white)))
print("MAE : {:.2f}".format(mean_absolute_error(white_y_test, y_predict_poly_white)))
print("MSE : {:.2f}".format(mean_squared_error(white_y_test, y_predict_poly_white)))
print("MAPE : {:.2f}".format(mean_absolute_percentage_error(white_y_test, y_predict_poly_white)))
