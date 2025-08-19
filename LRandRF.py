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

# Assuming 'X' = features,'y' = target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X = pd.get_dummies(df.drop('price', axis=1))
# df['column_name'] = df['column_name'].map({'yes': 1, 'no': 0}) 

# #Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# #Predict
# # y_predict_lr = lr.predict(X_test)

# #Evaluate
# print(f"Training R2 score: {model.score(X_train, y_train):.f} ")
# print(f"Test R2 score: {model.score(X_test, y_test):.f}")
