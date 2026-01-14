import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# Load the ARFF file
arff_file_path = 'gender_emodb_notch.arff'  # Replace with your ARFF file path
try:
    data, meta = arff.loadarff(arff_file_path)
except FileNotFoundError:
    print(f"Error: File '{arff_file_path}' not found. Please check the file path.")
    exit()

# Convert ARFF data to a Pandas DataFrame
df = pd.DataFrame(data)

# Ensure proper encoding of nominal attributes
for column in df.select_dtypes([object]).columns:
    df[column] = df[column].str.decode('utf-8')

# Separate features (X) and target variable (y)
X = df.iloc[:, :-1]  # All columns except the last one as features
y = df.iloc[:, -1]   # Last column as the target variable (class)

# Check for problematic values in numeric columns only
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
X_numeric = X[numeric_columns]

# Count rows before cleaning
rows_before = X_numeric.shape[0]

# Replace infinite values with NaN
X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
X_numeric.dropna(inplace=True)

# Count rows after cleaning
rows_after = X_numeric.shape[0]

# Calculate and print dropped rows
dropped_rows = rows_before - rows_after
print(f"Number of rows dropped due to NaN or infinite values: {dropped_rows}")

# Ensure y matches the cleaned X_numeric index
y = y[X_numeric.index]

# Scale the dataset to handle large values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Configure the base learner
random_tree = DecisionTreeClassifier(
    max_features=None,
    min_samples_split=2,
    min_impurity_decrease=0.001,
    random_state=1
)

# Initialize the Bagging classifier with the RandomTree as the base learner
bagging_classifier = BaggingClassifier(
    estimator=random_tree,
    n_estimators=100,              # Matches 100 iterations (-I 100)
    random_state=1,                # Matches -S 1
    n_jobs=1                       # Single-threaded execution (-num-slots 1)
)

# Perform 10-fold cross-validation and get predictions for all data points
cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
y_pred = cross_val_predict(bagging_classifier, X_scaled, y, cv=cv_splitter)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Calculate additional metrics
accuracy = accuracy_score(y, y_pred) * 100  # Convert to percentage for consistency with Weka output
kappa_statistic = (accuracy / 100 - (1 - accuracy / 100)) / (1 - (1 - accuracy / 100))  # Simplified kappa calculation
mean_abs_error = mean_absolute_error(pd.factorize(y)[0], pd.factorize(y_pred)[0])
root_mean_squared_error = np.sqrt(mean_squared_error(pd.factorize(y)[0], pd.factorize(y_pred)[0]))
relative_absolute_error = (mean_abs_error / np.mean(np.abs(pd.factorize(y)[0]))) * 100
root_relative_squared_error = (root_mean_squared_error / np.sqrt(np.mean(np.square(pd.factorize(y)[0])))) * 100

# Extract True Positives, False Positives, True Negatives, and False Negatives
TP = conf_matrix[1, 1]  # True Positives for the positive class
FP = conf_matrix[0, 1]  # False Positives for the positive class
TN = conf_matrix[0, 0]  # True Negatives for the negative class
FN = conf_matrix[1, 0]  # False Negatives for the positive class

# Calculate metrics
TP_rate = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Recall)
FP_rate = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP_rate  # Recall is the same as TP Rate
F_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0

#Calculate ROC Area and PRC Area using sklearn's roc_auc_score and average_precision_score
roc_area = roc_auc_score(pd.factorize(y)[0], pd.factorize(y_pred)[0]) if len(np.unique(y)) == 2 else None
prc_area = average_precision_score(pd.factorize(y)[0], pd.factorize(y_pred)[0]) if len(np.unique(y)) == 2 else None

# Print results

print(f"\nCorrectly Classified Instances         {int(accuracy / 100 * len(y))}               {accuracy:.4f} %")
print(f"Incorrectly Classified Instances         {len(y) - int(accuracy / 100 * len(y))}                {100 - accuracy:.4f} %")
print(f"Kappa statistic                          {kappa_statistic:.4f}")
print(f"Mean absolute error                      {mean_abs_error:.4f}")
print(f"Root mean squared error                  {root_mean_squared_error:.4f}")
print(f"Relative absolute error                  {relative_absolute_error:.4f} %")
print(f"Root relative squared error              {root_relative_squared_error:.4f} %")
print(f"Total Number of Instances              {len(y)}")


# Print metrics
print(f"                 TP Rate: {TP_rate:.3f}")
print(f"                 FP Rate: {FP_rate:.3f}")
print(f"                 Precision: {precision:.3f}")
print(f"                 Recall: {recall:.3f}")
print(f"                 F-Measure: {F_measure:.3f}")
print(f"                 MCC: {MCC:.3f}")
if roc_area is not None:
    print(f"                 ROC Area: {roc_area:.3f}")
if prc_area is not None:
    print(f"                 PRC Area: {prc_area:.3f}")

print("\n=== Confusion Matrix ===")
classes = np.unique(y)
for i, row in enumerate(conf_matrix):
    print(f"{' '.join(map(str, row))} |   {classes[i]}")

