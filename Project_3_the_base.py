# PROJECT 3 - BASIC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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


X = salary_data.iloc[:, :-1].values
y = salary_data.iloc[:, -1].values

# splitting datas into training and test datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)


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



