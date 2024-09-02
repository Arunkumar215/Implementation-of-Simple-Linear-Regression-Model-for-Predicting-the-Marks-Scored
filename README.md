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
