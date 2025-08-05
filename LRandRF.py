from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('Housing.csv')
X = df.drop('price', axis=1)
y = df['price']

# Assuming 'X' = features, 'y' = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train
lr = LinearRegression()
lr.fit(X_train, y_train)

#Predict
y_predict_lr = lr.predict(X_test)

#Evaluate
print("Linear Regression RMSE: ", np.sqrt(mean_squared_error(y_test, y_predict_lr)))
print("R2 Score: ", r2_score(y_test, y_predict_lr))
