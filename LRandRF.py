from sklearn.linear_model import LinearRegression
import pandas

#Train
lr = LinearRegression()
lr.fit(X_train, y_train)

#Predict
y_predict_lr = lr.predict(X_test)