# DiagnoSmart

**Pneumonia**

Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis. The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.

**Diabetes**

Diabetes is a chronic medical condition characterized by high blood sugar levels. The hormone insulin, produced by the pancreas, helps regulate blood sugar levels by allowing glucose to enter cells for energy. In people with diabetes, their bodies either do not produce enough insulin or do not use it properly, leading to elevated blood sugar levels.

There are two main types of diabetes:
1. Type 1 Diabetes: This type of diabetes is an autoimmune disease in which the immune system attacks and destroys the insulin-producing cells in the pancreas. This leads to a lack of insulin in the body, requiring insulin injections for survival.
2. Type 2 Diabetes: This type of diabetes is the most common and is often associated with lifestyle factors such as poor diet, lack of exercise, and obesity. In this type of diabetes, the body becomes resistant to the effects of insulin, leading to elevated blood sugar levels.

**Heart Disease**

Heart disease refers to a range of conditions that affect the heart and blood vessels. It is one of the leading causes of death worldwide. The most common type of heart disease is coronary artery disease, which is caused by a buildup of plaque in the arteries that supply blood to the heart.

Other types of heart disease include heart failure, arrhythmia (abnormal heart rhythm), heart valve disease, and congenital heart disease (present at birth).

Risk factors for heart disease include high blood pressure, high cholesterol, smoking, obesity, diabetes, family history, and a sedentary lifestyle. Symptoms of heart disease may vary depending on the type of heart disease, but can include chest pain or discomfort, shortness of breath, fatigue, lightheadedness, and heart palpitations.

Diagnosis of heart disease involves a physical exam, medical history, and various tests such as an electrocardiogram (ECG), echocardiogram, stress test, or cardiac catheterization. Treatment options for heart disease depend on the type and severity of the condition, but may include lifestyle changes such as diet and exercise, medications, medical procedures, or surgery.

**Motivation for the Project** 
-----------------------------------------------------------------
The motivation for multiple diseases prediction using machine learning lies in the potential to improve patient outcomes and reduce healthcare costs. By leveraging the power of machine learning algorithms, we can develop models that can accurately predict the likelihood of a patient developing multiple diseases, such as diabetes, heart disease, and cancer.

These models can help healthcare providers identify patients who are at high risk of developing these diseases, allowing for earlier intervention and more targeted preventative measures. For example, if a model predicts that a patient has a high likelihood of developing diabetes, healthcare providers can work with that patient to implement lifestyle changes such as diet and exercise to reduce their risk.

Machine learning models can also help with the early detection of diseases, which can lead to better treatment outcomes and lower healthcare costs. By analyzing patient data and identifying patterns, models can identify patients who may have early signs of a disease, even before symptoms appear. This allows for earlier diagnosis and treatment, potentially leading to better outcomes and reduced healthcare costs.

**Steps of the Project**
-----------------------------------------------------------------
**Data Collection:**

Collect data on symptoms, medical history, and other relevant factors for patients diagnosed with different diseases. This data can be collected through various sources such as electronic health records (EHR), patient surveys, or other medical records. We have collect our data from the mention sources:
1. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. https://www.kaggle.com/datasets/johndasilva/diabetes
3. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset


**Data Preprocessing:**

Process the collected data to clean and transform it into a suitable format for the model. This can include tasks such as:
1. Data cleaning 
2. Normalization 
3. Feature extraction
4. Feature selection.


**Feature Engineering:**

Identify relevant features from the preprocessed data that could help in predicting diseases. This can be done using techniques such as:
1. correlation analysis
2. Principal component analysis (PCA)
3. Other feature selection algorithms.


**Model Building:**

Build a machine learning model using the preprocessed and engineered features. There are several models that could be used for this, such as: 
1. Decision Tree
2. Random Forest
3. Logistic Regression
4. Support Vector Machines
5. Neural Networks
6. Convolutional Neural Network.


**Model Training:**

Train the machine learning model using the preprocessed data. Split the data into training and validation sets to evaluate the model's performance on new, unseen data.

Pneumonia Disease prediction model compiling 
![Pneumonia Disease prediction model compiling ](https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/2.png "Pneumonia Disease prediction model compiling ")


**Model Evaluation:**

Evaluate the model's performance on the validation set. Metrics such as accuracy, precision, recall, and F1 score can be used to evaluate the model's performance.

Training and Validation Accuracy in Pneumonia prediction
![Training and Validation Accuracy in Pneumonia prediction]( https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/3.png "Training and Validation Accuracy in Pneumonia prediction")

Evaluate for a person
![Evaluate for a person]( https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/4.png "Evaluate for a person")
Evaluate for multiple person
![Evaluate for multiple person]( https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/5.png "Evaluate for multiple person")

Accuracy in heart disease prediction Training data
![Accuracy in heart disease prediction Training data](https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/9.png "Accuracy in heart disease prediction Training data")
Accuracy in heart disease prediction Test data
![Accuracy in heart disease prediction Test data]( https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/10.png "Accuracy in heart disease prediction Test data")

Evaluate for a person
![Evaluate for a person]( https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/11.png "Evaluate for a person")


Accuracy in diabetes prediction Training data
![Accuracy in diabetes prediction Training data](https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/6.png "Accuracy in diabetes prediction Training data")
Accuracy in heart disease prediction Test data
![Accuracy in  diabetes prediction Test data]( https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/7.png "Accuracy in diabetes prediction Test data")

Evaluate for a person
![Evaluate for a person]( https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/8.png "Evaluate for a person")

**Hyperparameter Tuning:**

Optimize the model's hyperparameters to improve its performance on the validation set. This can be done using techniques such as:
1. Grid Search
2. Random Search
3. Bayesian Optimization.


**Model Deployment:**

Deploy the trained model into a production environment. This can be done using various technologies such as
1. Flask
2. Django
3. RESTful APIs.


**User Interface:**

Create a user interface that allows users to input their symptoms and medical history, and receive predictions for their likelihood of having a particular disease. This can be done using libraries such as streamlit, Tkinter or PyQt.

Pneumonia Prediction Page
![Pneumonia Prediction Page](https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/12.JPG "Pneumonia Prediction Page")
![Pneumonia Prediction Page1](https://github.com/NAYANCSE27/DiagnoSmart/blob/final/images/13.JPG "Pneumonia Prediction Page1")


**Testing and Maintenance:**

Test the system thoroughly and maintain it by updating the model and user interface as new data and features become available.

Overall, the project aims to build a machine learning model that can predict multiple diseases based on patient symptoms and medical history. The model is trained on a large dataset of patients diagnosed with various diseases and is evaluated on new, unseen data. The model is then deployed into a production environment, where users can input their symptoms and medical history to receive predictions for their likelihood of having a particular disease.
