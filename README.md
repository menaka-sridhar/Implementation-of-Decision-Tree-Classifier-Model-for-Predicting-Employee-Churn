# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Load the Dataset:Import the employee churn dataset using Python libraries.
2.Preprocess the Data:Convert categorical data into numerical format and separate features and target values.
3.Split the Dataset:Divide the dataset into training and testing sets.
4.Train the Decision Tree Classifier:Create and train the Decision Tree model using training data.
5.Predict and Evaluate:Predict employee churn and calculate model accuracy using test data
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Menaka M S
RegisterNumber:212225040232  
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
data = {
    "Age": [25, 30, 35, 40, 28, 32, 45, 50, 29, 38],
    "Salary": [30000, 40000, 50000, 60000, 35000,
               45000, 70000, 80000, 38000, 52000],
    "Years_at_Company": [1, 3, 5, 10, 2, 4, 12, 15, 2, 6],
    "Churn": [1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
}
df = pd.DataFrame(data)
print("Dataset:\n")
print(df)
X = df[["Age", "Salary", "Years_at_Company"]]
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
print("\nFeature Importance:")
print(importance)
plt.figure(figsize=(12,8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True
)
plt.title("Decision Tree for Employee Churn Prediction")
plt.show()
employee_data = [[30, 45000, 3]]
prediction = model.predict(employee_data)
print("\nCustom Employee Prediction:")
if prediction[0] == 1:
    print("Employee is Likely to Leave the Company")
else:
    print("Employee is Likely to Stay in the Company")
```
## Output:
![decision tree classifier model](sam.png)
<img width="327" height="232" alt="image" src="https://github.com/user-attachments/assets/1ea563df-2910-46aa-8157-907a7c6d609d" />
<img width="811" height="696" alt="image" src="https://github.com/user-attachments/assets/dcbae972-3c68-4870-83ce-0f7cfcf86040" />
<img width="296" height="36" alt="image" src="https://github.com/user-attachments/assets/39feac45-7c18-4148-8623-61a8312ebd9d" />
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
