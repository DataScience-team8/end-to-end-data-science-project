from sklearn.linear_model import LogisticRegression

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(white_X_train, white_y_train)

# Predict using the model
y_predict = model.predict(white_X_test)
# evaluation
print(model.score(white_X_test, white_y_test))

# confusion_matrix
df = pd.DataFrame({'y_predicted': y_predict,
                   'y_actual':white_y_test})
confusion_matrix = pd.crosstab(df['y_actual'],
df['y_predicted'], rownames=['Actual'],
colnames=['Predicted'])
print(confusion_matrix)

# Confusion matrix visualization
sn.heatmap(confusion_matrix, annot=True, fmt='.0f')
