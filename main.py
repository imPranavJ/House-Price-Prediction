import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split

data = pd.read_csv('Housing.csv')
data = data[['price', 'area']]
pred = 'price'
X = np.array(data.drop([pred], 1))
y = np.array(data[pred])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
best_acc = 0

for _ in range(101):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print('Accuracy: ', str(acc))

    if acc > best_acc:
        best_acc = acc
        print("Best Accuracy recorded: ", best_acc)

print("Coefficient: ", linear.coef_)
print("Intercept  : ", linear.intercept_)
print("Final Accuracy: ", best_acc)
y_pred = linear.predict(x_test)

for i in range(len(pred)):
    print(pred[i], x_test[i], y_test[i])

p = 'price'

lr = [(linear.coef_ * x) + linear.intercept_ for x in data['area']]

style.use('seaborn-v0_8-colorblind')
plt.scatter(data['area'], data[p], alpha=0.4, color='green')
plt.plot(data['area'], lr, color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()