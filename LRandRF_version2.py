from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Housing.csv')

# Drop unnecessary columns
columns_to_drop = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
df.drop(columns=columns_to_drop, inplace=True)

# Clean and encode target variable
df['prefarea'] = df['prefarea'].str.strip().str.lower()
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})

# One-hot encode categorical variable
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Features and target
X = df.drop('prefarea', axis=1)
y = df['prefarea']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate accuracy
print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# Predict probabilities for ROC curve
y_prob = model.predict_proba(X_test)[:, 1]

# Predict labels for confusion matrix
y_pred = model.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
