from imblearn.combine import SMOTEENN
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, fbeta_score, roc_auc_score, \
    average_precision_score, confusion_matrix

RANDOM_STATE = 20


def get_train_and_test_values(df_train, df_valid, df_test):
    X_train = df_train.drop('FraudFound_P', axis=1).values
    y_train = df_train['FraudFound_P'].values

    X_valid = df_valid.drop('FraudFound_P', axis=1).values
    y_valid = df_valid['FraudFound_P'].values

    X_test = df_test.drop('FraudFound_P', axis=1).values
    y_test = df_test['FraudFound_P'].values

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def with_resampling(df_train, df_valid, df_test):
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_and_test_values(df_train, df_valid, df_test)

    smote_enn = SMOTEENN(random_state=RANDOM_STATE, sampling_strategy=0.6)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled, X_valid, y_valid, X_test, y_test


def obtain_predictions(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def get_scores(y_test, predictions):
    precision = precision_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    fbeta = fbeta_score(y_test, predictions, beta=2)
    roc_auc = roc_auc_score(y_test, predictions)
    pr_auc = average_precision_score(y_test, predictions)

    print("Precision: %.3f" % precision)
    print("Accuracy: %.3f" % accuracy )
    print("Recall: %.3f" % recall)
    print("F1: %.3f" % f1)
    print("Fbeta: %.3f" % fbeta)
    print("ROC AUC: %.3f" % roc_auc)
    print("PR AUC: %.3f" % pr_auc )


def get_confusion_matrix(actual, predicted, title):
    cf = confusion_matrix(actual, predicted)
    group_names = ['True Negative','False Positive','False Negative','True Positive'] 
    group_counts = ['{0:0.0f}'.format(value) for value in cf.flatten()] 

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    s = sns.heatmap(cf, annot=labels, fmt='', cmap='Blues', xticklabels=['non fraud', 'fraud'], yticklabels=['non fraud', 'fraud'])
    s.set(xlabel='Predictions', ylabel='True Labels', title=title)
    return s