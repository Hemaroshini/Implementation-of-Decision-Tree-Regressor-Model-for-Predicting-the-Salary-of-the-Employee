# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.


## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: HEMAROSHINI M
RegisterNumber: 212219220015

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
Data Head:

![image](https://user-images.githubusercontent.com/107909531/174950374-61c6b95b-318d-432a-8d7d-df07f9470fc0.png)


Data Info:

![image](https://user-images.githubusercontent.com/107909531/174950415-460db5b3-6692-4c8a-b8fc-24026b023009.png)


Data Isnull:

![image](https://user-images.githubusercontent.com/107909531/174950460-6bc84b06-8c9b-42d9-9003-5583fe1b634a.png)


Data Head:

![image](https://user-images.githubusercontent.com/107909531/174950507-6fbf5ec8-4a70-4903-924e-065265ecc194.png)


MSE:

![image](https://user-images.githubusercontent.com/107909531/174950538-fed1c5be-d0d0-415a-a651-0b67d2518f74.png)


R2:

![image](https://user-images.githubusercontent.com/107909531/174950558-a7963291-3e20-43b1-9c8e-1d614a1735ed.png)


Predicted Value:

![image](https://user-images.githubusercontent.com/107909531/174950585-c8d60988-acfb-426d-a87a-f9fa403624f8.png)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
