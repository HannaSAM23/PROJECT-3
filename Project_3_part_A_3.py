# PROJECT 3 - A.3
# STOCHASTIC GRADIENT DESCENT FOR THE SAME DATASET

# Importing various packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
salary_data = pd.read_csv(r'C:\Users\hamdy\Downloads\Salary.csv')
salary_data = salary_data.rename(columns={'YearsExperience': 'years_experience', 'Salary': 'salary'})

X = salary_data.iloc[:, :-1].values
y = salary_data.iloc[:, -1].values

n = len(y)
X_b = np.c_[np.ones((n, 1)), X]

XT_X = X.T @ X
theta_linreg = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)
print("Theta from normal equation:", theta_linreg)

# Stochastic Gradient Descent
n_epochs = 50
t0, t1 = 5, 50
m = n # number of training examples


def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2.0* xi.T @ (xi @ theta-yi)
        eta = learning_schedule(epoch*m+i)
        theta -= eta*gradients
print("theta from own sdg")
print(theta)

# Plotting
x_range = np.linspace(X.min(), X.max(), 100)
X_range_b = np.c_[np.ones((100,1)), x_range]
y_predict_sgd = X_range_b.dot(theta)
y_predict_linreg = X_range_b.dot(theta_linreg)


plt.plot(X, y, 'ro', label='Data')
plt.plot(x_range, y_predict_sgd, "r-", label='SGD Prediction')
plt.plot(x_range, y_predict_linreg ,'b-', label='Normal Eq Prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience')
plt.legend()
plt.show()
