**❤️ Heart Disease Prediction using Machine Learning**

A simple and beginner-friendly Machine Learning project that predicts the risk of heart disease based on user-provided medical parameters.
This project implements multiple classification algorithms, compares their performance, and selects the best model for final prediction.

**📌 Project Overview**
Heart disease is one of the leading causes of death worldwide. Early detection can help doctors and patients take preventive measures.
This project uses machine learning to predict whether a person is at risk of heart disease based on health-related inputs like:

Age
Sex
Chest Pain Type
Blood Pressure
Cholesterol
Blood Sugar
ECG Results
Maximum Heart Rate
Exercise-Induced Angina
Oldpeak
Slope
Number of Major Vessels
Thalassemia

**🧠 Machine Learning Models Used**
The project applies and compares multiple classification algorithms:

Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier
Naive Bayes

After evaluation using Accuracy, Precision, Recall, F1-score, and AUC, the Decision Tree model performed the best and was selected for final predictions.

📊 Model Comparison Results
Model	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.860	0.847	0.782	0.813	0.950
KNN	0.795	0.760	0.692	0.724	0.839
SVM	0.895	0.880	0.846	0.862	0.973
Decision Tree	1.000	1.000	1.000	1.000	1.000
Random Forest	0.990	1.000	0.974	0.987	1.000
Gradient Boosting	1.000	1.000	1.000	1.000	1.000
Naive Bayes	0.925	0.943	0.858	0.899	0.985


**🥇 Best Model**
✔ Decision Tree Classifier
Achieved perfect performance across all metrics.
Selected as the final model for predictions.

**🚀 Features**
Clean and structured code
Uses multiple classification algorithms
Automatic model comparison
User input-based prediction
Easy to understand for beginners
Suitable for learning data preprocessing, model evaluation, and ML workflows

**🧪 Tech Stack**
Python
Pandas, NumPy
Scikit-learn
Matplotlib / Seaborn (optional)
Jupyter Notebook / VS Code


📢 Project Workflow
Data Loading
Data Preprocessing
Handling Missing Values
Encoding (Label Encoding / One-Hot Encoding)
Feature Scaling
Train-Test Split
Model Training
Model Comparison
Best Model Selection
Final Prediction
