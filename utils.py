def obtain_predictions(model, X_train, y_train, X_test, y_test):
    model.fit(X_train ,y_train)
    predictions = model.predict(X_test)
    return predictions


def get_scores(y_test, predictions):
    precision = precision_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    fbeta = fbeta_score(y_test, predictions, beta=0.75)
    roc_auc = roc_auc_score(y_test, predictions)
    pr_auc = average_precision_score(y_test, predictions)

    print("Precision: %.2f%%" % (precision * 100.0))
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Recall: %.2f%%" % (recall * 100.0))
    print("F1: %.2f%%" % (f1 * 100))
    print("Fbeta: %.2f%%" % (fbeta * 100))
    print("ROC AUC: %.2f%%" % (roc_auc * 100))
    print("PR AUC: %.2f%%" % (pr_auc * 100))
