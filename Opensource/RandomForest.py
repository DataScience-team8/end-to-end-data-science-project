from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

cv_accuracy =[]
n_iter =0

for train_idx, test_idx in Sk.split(white_X,white_y):
  # Train the Random Forest Classifier model
  model = RandomForestClassifier()
  model.fit(white_X_train,white_y_train)

  # Predict using the model
  y_predict=model.predict(white_X_test)

  # evaluation
  print(accuracy_score(white_y_test,y_predict))
  accuracy = np.round(accuracy_score(white_y_test,y_predict),4)
  cv_accuracy.append(accuracy)
print(f"mean score: {np.mean(cv_accuracy)}")
# confusion_matrix
df = pd.DataFrame({'y_predicted': y_predict,
                   'y_actual': white_y_test})
confusion_matrix = pd.crosstab(df['y_actual'],
                               df['y_predicted'], rownames=['Actual'],
                               colnames=['Predicted'])
print(confusion_matrix)
# confusion_matrix visualization
sn.heatmap(confusion_matrix, annot=True, fmt='.0f')
