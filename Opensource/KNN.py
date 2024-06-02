from sklearn.neighbors import KNeighborsClassifier

cv_accuracy =[]
n_iter =0

for train_idx, test_idx in Sk.split(white_X,white_y):
  # Train the KNN model
  knn = KNeighborsClassifier(n_neighbors = 10)
  knn.fit(white_X_train,white_y_train)

  # Predict using the model
  y_predict = knn.predict(white_X_test)

  #evaluation
  print(knn.score(white_X_test, white_y_test))
  accuracy =np.round(knn.score(white_X_test, white_y_test),4)
  cv_accuracy.append(accuracy)
print(f"\nmean score: {np.mean(cv_accuracy)}")
# confusion_matrix
df = pd.DataFrame({'y_predicted': y_predict,
                   'y_actual':white_y_test})
confusion_matrix = pd.crosstab(df['y_actual'],
df['y_predicted'], rownames=['Actual'],
colnames=['Predicted'])
print(confusion_matrix)

# confusion_matrix visualization
sn.heatmap(confusion_matrix, annot=True, fmt='.0f')
