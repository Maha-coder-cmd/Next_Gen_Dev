from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Housing.csv')
columns_to_drop = ['mainroad','guestroom','basement','hotwaterheating','airconditioning']
df_clean = df.drop(columns = columns_to_drop)

X = df.drop('price', axis=1)
y = df['price']

df.head()

print(df.head())

df['prefarea'] = df ['prefarea'].str.strip().str.lower()

df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'])

X = df.drop('prefarea', axis=1)
y = df['prefarea']

# Assuming 'X' = features,'y' = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("how many values => ", train_test_split(X, y, test_size=0.2, random_state=42))

# print(df['prefarea'].unique())

# X_train['prefarea'] = X_train['prefarea'].map({'yes': 1,'no': 0})
# X_test ['prefarea'] = X_test['prefarea'].map({'yes': 1, 'no': 0})

print("Data types in X_train:")
print(X_train.dtypes)

# Show first few rows to see raw data
print("First 5 rows of X_train:")
print(X_train.head())

# print(df['furnishingstatus'].unique())



#Train model
model = LinearRegression()
model.fit(X_train, y_train)

# #Predict
# # y_predict_lr = lr.predict(X_test)

# #Evaluate
# print(f"Training R2 score: {model.score(X_train, y_train):.f} ")
# print(f"Test R2 score: {model.score(X_test, y_test):.f}")
