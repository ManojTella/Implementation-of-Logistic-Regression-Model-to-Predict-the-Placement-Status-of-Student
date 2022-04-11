# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import pandas module.
2. Read the required csv file using pandas.
3. Import LabEncoder module.
4. From sklearn import logistic regression.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. print the required values.
8. End the program. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Manoj Guna Sundar Tella.
RegisterNumber:  212221240026.
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output:

![img1](https://user-images.githubusercontent.com/94883876/162769486-ecc12f9a-93ee-43c0-b8e4-b01f259878b0.png)
![img2](https://user-images.githubusercontent.com/94883876/162769564-bbd48196-8593-441d-b330-fdbe11103d8d.png)
![img3](https://user-images.githubusercontent.com/94883876/162769588-fdaa4ac3-2f4c-4407-923c-2fc09857f24d.png)
![img4](https://user-images.githubusercontent.com/94883876/162769610-4220d619-c6a1-44f4-9cd7-455a6b46ec1d.png)
![img5](https://user-images.githubusercontent.com/94883876/162769640-f509e420-b67f-4651-a360-3464f4eb6991.png)
![img6](https://user-images.githubusercontent.com/94883876/162769655-a9116a32-d226-4bae-b604-78ffabf7fde0.png)
![img7](https://user-images.githubusercontent.com/94883876/162769672-62fbda43-4255-4b88-bc68-c7e51581beec.png)
![img8](https://user-images.githubusercontent.com/94883876/162769689-5151a17e-7ac2-4509-a5dc-1ea60c13539c.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
