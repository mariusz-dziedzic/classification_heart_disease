# Import libraries:
# Model import Logistic Regression
from sklearn.linear_model import LogisticRegression
# Import of a prediction accuracy measure
from sklearn.metrics import accuracy_score
# Load the function that divides the file into training and test data
from sklearn.model_selection import train_test_split
# Import of the function responsible for generating the confusion matrix
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Upload data:
df = pd.read_csv('heart.csv')
df.head()

# X is the variable that contains the data that will be used for prediction
X = df.iloc[:, :-1].values
# Y is the variable that will be tested and predicted
y = df.iloc[:, -1].values
# Division into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model building
model = LogisticRegression(solver='lbfgs', max_iter=1000)
# Model training
model.fit(X_train, y_train)
# Prediction based on the model
y_predicted = model.predict(X_test)
confusion_matrix(y_predicted, y_test)

# Accuracy - (TN+TP)/N
# Percentage of correct classifications
accuracy = accuracy_score(y_test, y_predicted)
accuracy = round(accuracy, 3) * 100
print('Accuracy:', accuracy, '%')

# Overall Error Rate - (FN+FP)/N
# Percentage of misclassifications
oer = 1 - accuracy_score(y_predicted, y_test)
oer = round(oer, 3) * 100
print('OER:', oer, '%')

# Sensitivity, Recall – (TP/(FN+TP)
# Percentage of correctly classified positive cases
recall = recall_score(y_predicted, y_test)
recall = round(recall, 3) * 100
print('Recall:', recall, '%')

# Precision – TP/(FP+TP)
# Percentage of correct classifications among all those classified as positive
precision = precision_score(y_predicted, y_test)
precision = round(precision, 3) * 100
print('Precision:', precision, '%')

plot_roc_curve(model, X_test, y_test)
# Show AUC on graph:
# plt.show()  # If you want to see,  uncomment this line
# In orginal model, AUC = 0.89. It is an good result

# Define one new data instance (One patient's data)
# Heading:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca
# Example: Xnew = [[60,1,4,117,230,1,0,160,1,1.4,1,2]]

while True:
    decision = input("Do you want to enter patient data? [Y/N]: ")
    if decision not in ["Y", "y", "yes"]:
        break
    try:
        age = int(input("Age in years: "))
        sex = int(input("Sex (1 = male; 0 = female): "))
        cp = int(input("Chest pain type (0, 1, 2, 3, 4): "))
        trestbps = int(input("Resting blood pressure (in mm Hg on admission): "))
        chol = int(input("Serum cholesterol in mg/dl: "))
        fbs = int(input("fasting blood sugar in 120 mg/dl (1 = true; 0 = false): "))
        restecg = int(input("Resting electrocardiogram results (0, 1, 2): "))
        thalach = int(input("Maximum heart rate achieved: "))
        exang = int(input("Exercise induced angina (1 = yes; 0 = no): "))
        oldpeak = float(input("ST depression induced by exercise relative to rest: "))
        slope = int(input("Slope of the peak exercise ST segment: "))
        ca = int(input("Number of major vessels (0-3) colored by fluoroscopy: "))

        Xnew = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]]

        # Make a prediction
        ynew = model.predict(Xnew)
        # Show the inputs and predicted outputs
        print()
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
        print()
        if ynew[0]:
            print("There is a high risk of Heart Disease ")
        else:
            print("There is a small risk of heart disease")

        add_decision = input("Do you want to add the results to the database? [Y/N]: ")
        if add_decision not in ["Y", "y", "yes"]:
            break
        try:
            Xnew[0].append(ynew[0])
            with open(r'heart.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(Xnew[0])
        except:
            print("The entry could not be added to the database ")
    except:
        print("You only need to enter numeric values ")
        print("Incorrect result ")
input()
