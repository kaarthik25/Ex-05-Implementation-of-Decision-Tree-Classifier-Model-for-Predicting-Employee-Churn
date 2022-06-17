# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.import pandas module and import the required data set.</br>
2.Find the null values and count them.</br>
3.Count number of left values.</br>
4.From sklearn import LabelEncoder to convert string values to numerical values.</br>
5.From sklearn.model_selection import train_test_split.</br>
6.Assign the train dataset and test dataset.</br>
7.From sklearn.tree import DecisionTreeClassifier.</br>
8.Use criteria as entropy.</br>
9.From sklearn import metrics.</br>
10.Find the accuracy of our model and predict the require values.</br>

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kaarthikeyan.S
RegisterNumber: 212220040068 
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
 

## Output:
![d1](https://user-images.githubusercontent.com/94525701/169461369-b612f99f-9276-4a8f-9154-42e0e96fc419.png)
![d2](https://user-images.githubusercontent.com/94525701/169461392-f865099a-1b17-45ef-b36b-c2a790181dd1.png)
![d3](https://user-images.githubusercontent.com/94525701/169461403-b7f6d52e-30a0-41ab-8b31-4807625a7bc2.png)
![d4](https://user-images.githubusercontent.com/94525701/169461410-cd669d83-f685-4e38-9a70-f38a1a95455f.png)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
