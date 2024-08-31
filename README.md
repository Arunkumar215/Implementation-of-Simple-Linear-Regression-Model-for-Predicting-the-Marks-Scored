# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Arunkumar S A
RegisterNumber:  212223220009
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
![image](https://github.com/user-attachments/assets/1a6ade56-a40d-4e39-aa31-8486eb8c1b93)
![image](https://github.com/user-attachments/assets/e9fe0e31-b8e8-4219-b775-1c11a9774ac0)
![image](https://github.com/user-attachments/assets/16752cef-ce04-483c-ad64-85ba43258044)
![image](https://github.com/user-attachments/assets/ac9ee500-14b7-40d1-8afe-3c1d6392cf69)
![image](https://github.com/user-attachments/assets/8578347e-0d2b-4261-8aa6-4372e6193e7c)
![image](https://github.com/user-attachments/assets/4f8742fd-9722-401f-8cb6-f356a2f0c2ba)
![image](https://github.com/user-attachments/assets/3ef609dd-19fa-4e29-8df1-bb438e9ff1a1)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
