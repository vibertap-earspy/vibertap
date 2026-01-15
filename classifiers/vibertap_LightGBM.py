#!/usr/bin/env python
# coding: utf-8

# # EarSpy LightGBM Classification Notebook
# 
# This notebook trains and evaluates a **LightGBM** classifier on the EarSpy feature dataset (`rls_7t.csv`).  
# 
# It assumes:
# - The CSV file `rls_7t.csv` is in the same directory as this notebook.
# - There is **no header row** in the CSV.
# - The **last column** contains the class label (e.g., `zero`, `one`, ...).
# - All previous columns are numeric features (time-domain + frequency-domain statistics).
# 

# In[53]:


# If lightgbm is not installed in your environment, uncomment and run this cell once
# !pip install lightgbm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb

import matplotlib.pyplot as plt
import itertools


# In[ ]:





# In[ ]:





# ## 1. Load and inspect the dataset

# In[54]:


# Path to your CSV file
csv_path = "gender_emodb_clmap.csv"

# Load the dataset with no header row
df = pd.read_csv(csv_path, header=None)

print("Data shape:", df.shape)
display(df.head())

print("\nColumn indices:", df.columns.tolist())


# ## 2. Preprocess: features and labels

# In[55]:


# Last column is the label
label_col = df.columns[-1]
feature_cols = df.columns[:-1]

print("Label column index:", label_col)
print("Number of feature columns:", len(feature_cols))

# Features and raw labels
X = df[feature_cols].values
y_raw = df[label_col].values

print("Sample raw labels:", y_raw[:10])


# ### 2.1 Encode string labels to integers

# In[56]:


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("Encoded classes:", label_encoder.classes_)
print("Encoded labels sample:", y[:10])
num_classes = len(label_encoder.classes_)


# ## 3. Train–test split

# In[57]:


x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)


# ## 4. Define and train the LightGBM model (sklearn API)

# In[58]:


# Use the sklearn-style classifier API
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=num_classes,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)

# Train with evaluation set to monitor performance
lgb_model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    eval_metric='multi_logloss',
)

print("Model trained.")


# ## 5. Evaluate on the test set

# In[59]:


# Predict
y_pred = lgb_model.predict(x_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%\n")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# ## 6. Confusion matrix

# In[60]:


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
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix (LightGBM)')
plt.show()

# Plot normalized confusion matrix
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix (LightGBM, normalized)')
plt.show()


# ## 7. Feature importance

# In[61]:


# Get feature importance from LightGBM
importances = lgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature importances (LightGBM)")
plt.bar(range(len(feature_cols)), importances[indices])
plt.xticks(range(len(feature_cols)), [str(feature_cols[i]) for i in indices], rotation=90)
plt.tight_layout()
plt.show()

print("Feature importances (sorted):")
for idx in indices:
    print(f"Feature {feature_cols[idx]}: {importances[idx]:.4f}")


# ## 8. Next steps / tuning ideas
# 
# - Tune LightGBM hyperparameters:
#   - `n_estimators`, `num_leaves`, `max_depth`, `learning_rate`
#   - `min_child_samples`, `reg_alpha`, `reg_lambda`
# - Use stratified K-fold cross-validation to get more stable performance estimates.
# - Compare with other models on the same train–test split:
#   - XGBoost
#   - CatBoost
#   - RandomForest
#   - SVM
# - Save and load the trained model using `lgb_model.booster_.save_model("model.txt")`
#   and `lgb.Booster(model_file="model.txt")`.
# 

# In[ ]:




