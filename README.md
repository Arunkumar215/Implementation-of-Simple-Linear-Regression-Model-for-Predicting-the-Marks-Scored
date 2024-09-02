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
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("student_scores.csv") 
df.head()
df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="orangered",s=60)
plt.plot(x_train,regressor.predict(x_train),color="darkviolet",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()

plt.scatter(x_test,y_test,color="seagreen",s=60)
plt.plot(x_test,regressor.predict(x_test),color="cyan",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()


mse=mean_squared_error(_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
![image](https://github.com/user-attachments/assets/73511592-e734-408c-b938-e6dafdce443c)
![image](https://github.com/user-attachments/assets/1d611005-1971-470b-82df-cb6a90a2ad85)
![image](https://github.com/user-attachments/assets/0e7664b2-ba80-40f6-bff2-53628c042fd1)
![image](https://github.com/user-attachments/assets/589f94c2-dd76-4e23-af8e-2f3277c7e58a)
![image](https://github.com/user-attachments/assets/2304a966-acc0-4ad9-9e00-ce013876cd43)
![image](https://github.com/user-attachments/assets/c59812ff-59cb-4de0-9fa3-3e9f66a884f2)
![image](https://github.com/user-attachments/assets/bfc533d8-8788-4434-8385-b5b28f15ad02)
![image](https://github.com/user-attachments/assets/8a948422-4bd3-42f2-8c30-5e3560b0aecb)
![image](https://github.com/user-attachments/assets/af277aec-c0d3-4570-8c33-216b2114bdf8)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
