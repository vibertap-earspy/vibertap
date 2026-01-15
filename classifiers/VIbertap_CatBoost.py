#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import itertools



# Path to CSV file
csv_path = "gender_emodb_clmap.csv"


df = pd.read_csv(csv_path, header=None)

print("Data shape:", df.shape)
display(df.head())

print("\nColumn indices:", df.columns.tolist())




label_col = df.columns[-1]
feature_cols = df.columns[:-1]

print("Label column index:", label_col)
print("Number of feature columns:", len(feature_cols))

# Features and raw labels
X = df[feature_cols].values
y_raw = df[label_col].values

print("Sample raw labels:", y_raw[:10])




label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("Encoded classes:", label_encoder.classes_)
print("Encoded labels sample:", y[:10])
num_classes = len(label_encoder.classes_)




x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)




# Initialize CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=500,           # number of boosting rounds
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    random_seed=42,
    verbose=50                # print progress every 50 iterations
)

# Train the model
cat_model.fit(
    x_train, y_train,
    eval_set=(x_test, y_test),
    use_best_model=True
)

print("Model trained.")


# Evaluate on the test set




# Predict
y_pred = cat_model.predict(x_test)
# CatBoost returns shape (n_samples, 1), so flatten
y_pred = y_pred.flatten().astype(int)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%\n")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=9,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = label_encoder.classes_

# Plot non-normalized confusion matrix
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix (CatBoost)')
plt.show()

# Plot normalized confusion matrix
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix (CatBoost, normalized)')
plt.show()




# Get feature importance from CatBoost
importances = cat_model.get_feature_importance()
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature importances (CatBoost)")
plt.bar(range(len(feature_cols)), importances[indices])
plt.xticks(range(len(feature_cols)), [str(feature_cols[i]) for i in indices], rotation=90)
plt.tight_layout()
plt.show()

print("Feature importances (sorted):")
for idx in indices:
    print(f"Feature {feature_cols[idx]}: {importances[idx]:.4f}")




