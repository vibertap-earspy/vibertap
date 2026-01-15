

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import itertools




csv_path = "gender_emodb_clmap.csv"
df = pd.read_csv(csv_path, header=None)

print("Data shape:", df.shape)
display(df.head())


# Last column contains labels
label_col = df.columns[-1]

# All previous columns are features
feature_cols = df.columns[:-1]

print("Number of features:", len(feature_cols))
print("Label column:", label_col)


X = df[feature_cols].values
X = np.asarray(X, dtype=np.float64)


f32_max = np.finfo(np.float32).max
f32_min = -f32_max

mask_pos_inf = np.isposinf(X)
mask_neg_inf = np.isneginf(X)
mask_too_big = X > f32_max
mask_too_small = X < f32_min

num_changed = (
    mask_pos_inf.sum()
    + mask_neg_inf.sum()
    + mask_too_big.sum()
    + mask_too_small.sum()
)

X[mask_pos_inf] = f32_max
X[mask_neg_inf] = f32_min
X = np.clip(X, f32_min, f32_max)

print(f"[XGBoost sanitize] total entries capped/replaced: {int(num_changed)}")

y_raw = df[label_col].values






import numpy as np


X = np.asarray(X, dtype=np.float64)  

f32_max = np.finfo(np.float32).max   
f32_min = np.finfo(np.float32).min   

# Count how many entries are problematic
mask_pos_inf = np.isposinf(X)
mask_neg_inf = np.isneginf(X)
mask_too_big = X > f32_max
mask_too_small = X < f32_min

n_pos_inf   = int(mask_pos_inf.sum())
n_neg_inf   = int(mask_neg_inf.sum())
n_too_big   = int(mask_too_big.sum())
n_too_small = int(mask_too_small.sum())

# Replace / clip (NO dropping)
X[mask_pos_inf] = f32_max
X[mask_neg_inf] = f32_min
X = np.clip(X, f32_min, f32_max)

print("Sanitize report:")
print(f"  +inf replaced: {n_pos_inf}")
print(f"  -inf replaced: {n_neg_inf}")
print(f"  >float32 max clipped: {n_too_big}")
print(f"  <float32 min clipped: {n_too_small}")
print(f"  Total changed: {n_pos_inf + n_neg_inf + n_too_big + n_too_small}")





from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_raw)

print("Encoded classes:", label_encoder.classes_)




x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)





num_classes = len(np.unique(y))


n_classes = np.unique(y_train).size
if n_classes < 2:
    raise ValueError(f"y_train has only {n_classes} class. Check gender_7t labels after loading/encoding.")

common_params = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist',
    random_state=42
)

if n_classes == 2:
    xgb_model = XGBClassifier(
        **common_params,
        objective='binary:logistic',
        eval_metric='logloss'
    )
else:
    xgb_model = XGBClassifier(
        **common_params,
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=n_classes
    )

xgb_model.fit(x_train, y_train)
print("Model trained.")





# Predict
y_pred = xgb_model.predict(x_test)

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
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix (XGBoost)')
plt.show()

# Plot normalized confusion matrix
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix (XGBoost)')
plt.show()




# Plot feature importance from XGBoost
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature importances (XGBoost)")
plt.bar(range(len(feature_cols)), importances[indices])
plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Also print them sorted
print("Feature importances (sorted):")
for idx in indices:
    print(f"{feature_cols[idx]}: {importances[idx]:.4f}")













