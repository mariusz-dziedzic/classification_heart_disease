Before starting:
- The program is available in a version to run in Jupyter Notebook and in a standard Python file.
- Make sure you have all packages necessary for booting installed. They should be installed by default in Jupiter, but when running a script from a Python file, you may get an error related to importing the library. In this case, execute the following commands in CMD:
pip install scikit-learn
pip install pandas
pip install matplotlib 

Description:
- The program uses the machine learning technique to create a model that will be able to determine whether a patient with the given symptoms may have heart disease (determines the risk of its occurrence). Based on the available data (csv file), the model divides the data into 2 samples: the learning sample (80%) and the test sample (20%). The learned model then makes predictions on the data.
- Confusion matrix is ​​generated as well as parameters such as Accuracy, Sensitivity, Precision and Overall Error Rate.
- An AUC curve is also generated, which is a measure of the quality of the classification (result description below).
- Finally, the user can enter the patient's symptoms (according to a predefined pattern), and then he will receive information whether the patient may have heart disease (with some risk of measurement error) or not. Then the user can add the patient data automatically to the csv file. Then the user can test another patient and add data to the database.
- The next time the script is run, the model is re-evaluated with the new patient data, which should improve the accuracy of the diagnoses made by the model. 


##########################################################
The dataset Our dataset is provided by the Cleveland Clinic Foundation for Heart Disease. It's a CSV file with 303 rows (Orginal file). Each row contains information about a patient (a sample), and each column describes an attribute of the patient (a feature). We use the features to predict whether a patient has a heart disease (binary classification).
The description comes from the website: https://keras.io/examples/structured_data/structured_data_classification_from_scratch/ 


##########################################################
Table description:

Column	|			Description			|	Feature Type		|
------	|			----------			|	------------		|
Age	| Age in years 						| Numerical			|
Sex	| (1 = male; 0 = female)				| Categorical			|
CP 	| Chest pain type (0, 1, 2, 3, 4)			| Categorical			|
Trestbpd| Resting blood pressure (in mm Hg on admission)	| Numerical			|
Chol 	| Serum cholesterol in mg/dl 				| Numerical			|
FBS 	| fasting blood sugar in 120 mg/dl (1 = true; 0 = false)| Categorical			|
RestECG | Resting electrocardiogram results (0, 1, 2) 		| Categorical			|
Thalach | Maximum heart rate achieved 				| Numerical			|
Exang 	| Exercise induced angina (1 = yes; 0 = no) 		| Categorical			|
Oldpeak | ST depression induced by exercise relative to rest 	| Numerical			|
Slope 	| Slope of the peak exercise ST segment 		| Numerical			|
CA 	| Number of major vessels (0-3) colored by fluoroscopy 	| Both numerical & categorical	|
Thal 	| 3 = normal; 6 = fixed defect; 7 = reversible defect 	| Categorical			|
Target 	| Diagnosis of heart disease (1 = true; 0 = false) 	| Target			|


##########################################################
For the performed classification, we create a confusion matrix:
theoretical - columns
real - rows

|real\theoret|	   0|	  1|
|	    -|	   -|	  -|
|	    0|	 #TN|	#FP|
|	    1|	 #FN|   #TP|

Legend:
TN - True Negative -> Data classified correctly
FP - False Positive -> In real are negative
FN - False Negative -> In real are positive
TP - True Positive -> Data classified correctly
N - Sample Size


##########################################################
AUC:
The Area Under Curve (AUC) can be taken as a measure of the quality of the classification. AUC values and the quality of classification:
- AUC ∈ (0.6; 0.7⟩ - weak,
- AUC ∈ (0.7; 0.8⟩ - acceptable,
- AUC ∈ (0.8; 0.9⟩ - good,
- AUC ∈ (0.9; 1⟩ - excellent.



