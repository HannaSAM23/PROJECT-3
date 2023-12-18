# PROJECT 3 - PART A.1
# MAKING LIN REG WITH LASSO AND RIDGE
# INCLUDING MSE AND R2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
salary_data = pd.read_csv(r'C:\Users\hamdy\Downloads\Salary.csv')
salary_data.head()
salary_data.tail()
salary_data = salary_data.rename(columns={'YearsExperience': 'years_experience', 'Salary': 'salary'})
salary_data.columns
salary_data.shape
salary_data.describe()

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

X = salary_data.iloc[:, :-1].values
y = salary_data.iloc[:, -1].values

# splitting datas into training and test datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)

p = 1

I = np.eye(p,p)
# Decide which values of lambda to use
nlambdas = 100
MSEPredict = np.zeros(nlambdas)
MSETrain = np.zeros(nlambdas)
MSELassoPredict = np.zeros(nlambdas)
MSELassoTrain = np.zeros(nlambdas)
R2Predict = np.zeros(nlambdas)
R2Train = np.zeros(nlambdas)
R2LassoPredict = np.zeros(nlambdas)
R2LassoTrain = np.zeros(nlambdas)
lambdas = np.logspace(-4, 4, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    Ridgebeta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ y_train
    # include lasso using Scikit-Learn
    RegLasso = linear_model.Lasso(lmb)
    RegLasso.fit(X_train,y_train)
    # and then make the prediction
    ytildeRidge = X_train @ Ridgebeta
    ypredictRidge = X_test @ Ridgebeta
    ytildeLasso = RegLasso.predict(X_train)
    ypredictLasso = RegLasso.predict(X_test)
    MSEPredict[i] = MSE(y_test,ypredictRidge)
    MSETrain[i] = MSE(y_train,ytildeRidge)
    MSELassoPredict[i] = MSE(y_test,ypredictLasso)
    MSELassoTrain[i] = MSE(y_train,ytildeLasso)
    R2Predict[i] = R2(y_test,ypredictRidge)
    R2Train[i] = R2(y_train,ytildeRidge)
    R2LassoPredict[i] = R2(y_test,ypredictLasso)
    R2LassoTrain[i] = R2(y_train,ytildeLasso)

# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSETrain, label = 'MSE Ridge train')
plt.plot(np.log10(lambdas), MSEPredict, 'g--', label = 'MSE Ridge Test')
plt.plot(np.log10(lambdas), MSELassoTrain, label = 'MSE Lasso train')
plt.plot(np.log10(lambdas), MSELassoPredict, 'r--', label = 'MSE Lasso Test')
plt.title('LASSO vs. RIGDE reg. for MSE for training and data')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.log10(lambdas), R2Train, label = 'R2 Ridge train')
plt.plot(np.log10(lambdas), R2Predict, 'g--', label = 'R2 Ridge Test')
plt.plot(np.log10(lambdas), R2LassoTrain, label = 'R2 Lasso train')
plt.plot(np.log10(lambdas), R2LassoPredict, 'r--', label = 'R2 Lasso Test')
plt.title('LASSO vs. RIGDE reg. for R2 for training and data')
plt.xlabel('log10(lambda)')
plt.ylabel('R2')
plt.legend()
plt.show()


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Years of Experience (Training Set)')
plt.ylabel('Salary')
plt.xlabel('Years of Experience')
plt.show()

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Years of Experience (Test Set)')
plt.ylabel('Salary')
plt.xlabel('Years of Experience')
plt.show()



