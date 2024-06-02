from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
Sk = StratifiedKFold(n_splits=5,shuffle =True,random_state=42)

cv_accuracy =[]
macro_f1_mean= []
weighted_f1_mean= []
n_iter =0

for train_idx, test_idx in Sk.split(white_X,white_y):

  # Train the Decision Tree Classifier model
  model = DecisionTreeClassifier()
  model.fit(white_X_train, white_y_train)

  # Predict using the model
  y_pred = model.predict(white_X_test)
  n_iter +=1
  accuracy =np.round(accuracy_score(white_y_test,y_pred),4)
  train_size =white_X_train.shape[0]
  test_size =white_X_test.shape[0]
  print(f"iteration: {n_iter}, train size: {train_size},test size: {test_size}")

  cv_accuracy.append(accuracy)
  # evaluation
  # 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
  precision = precision_score(white_y_test, y_pred, average='macro')
  recall = recall_score(white_y_test, y_pred, average='macro')
  f1 = f1_score(white_y_test, y_pred, average='macro')
  macro_f1_mean.append(f1)
  # print evaluation
  print("\nmacro")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-score: {f1}")

  # Evaluation
  # 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
  precision = precision_score(white_y_test, y_pred, average='weighted')
  recall = recall_score(white_y_test, y_pred, average='weighted')
  f1 = f1_score(white_y_test, y_pred, average='weighted')
  weighted_f1_mean.append(f1)

  # print evaluation
  print("\nweighted")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-score: {f1}")
print(f"\nmean score: {np.mean(cv_accuracy)}, macro f1 mean score: {np.mean(macro_f1_mean)}, weighted f1 mean score: {np.mean(weighted_f1_mean)}")
