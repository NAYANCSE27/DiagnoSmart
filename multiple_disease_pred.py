# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 12:51:52 2023

@author: Dell
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open(
    './saved_file/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(
    './saved_file/heart_disease_model.sav', 'rb'))


# sidebar for navigate

with st.sidebar:

    selected = option_menu('Multiple Disease Prediction System Using Machine Learning',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Pneumonia Prediction'],

                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Disease Prediction Page
if (selected == 'Diabetes Prediction'):

    # page title
    st.title('Diabetes Disease Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)

    with col1:
        highBP = st.text_input('HighBP')

    with col2:
        highChol = st.text_input('HighChol')

    with col3:
        cholCheck = st.text_input('CholCheck')

    with col1:
        bmi = st.text_input('BMI')

    with col2:
        smoker = st.text_input('Smoker')

    with col3:
        stroke = st.text_input('Stroke')

    with col1:
        heartDiseaseorAttack = st.text_input('Heart Disease or Attack')

    with col2:
        physActivity = st.text_input('Physical Activity')

    with col3:
        fruits = st.text_input('Fruits')

    with col1:
        veggies = st.text_input('Veggies')

    with col2:
        heavyAlcoholConsump = st.text_input('Heavy Alcohol Consump')

    with col3:
        anyHealthCare = st.text_input('Any Health Care')

    with col1:
        noDocbcCost = st.text_input('NoDocbcCost')

    with col2:
        generalHealth = st.text_input('General Health')

    with col3:
        mentalHealth = st.text_input('Mental Health')

    with col1:
        physicalHealth = st.text_input('Physical Health')

    with col2:
        diffWalk = st.text_input('Difference Walk')

    with col3:
        sex = st.text_input('Sex')

    with col1:
        age = st.text_input('Age')

    with col2:
        education = st.text_input('Education')

    with col3:
        income = st.text_input('Income')

    # code for Prediction
    diabetic_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetic Disease Test Result'):
        diabetes_prediction = diabetes_model.predict(
            [[highBP, highChol, cholCheck, bmi, smoker, stroke, heartDiseaseorAttack,
              physActivity, fruits, veggies, heavyAlcoholConsump, anyHealthCare, noDocbcCost,
              generalHealth, mentalHealth, physicalHealth, diffWalk, sex, age,
              education, income]])

        if (diabetes_prediction[0] == 1):
            diabetic_diagnosis = 'The person is having diabetes disease'
        else:
            diabetic_diagnosis = 'The person does not have any diabetes disease'

    st.success(diabetic_diagnosis)


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    # page title
    st.title('Heart Disease Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)

    with col1:
        bmi = st.text_input('BMI')

    with col2:
        smoking = st.text_input('Smoking')

    with col3:
        alcoholDrinking = st.text_input('Alcohol Drinking')

    with col1:
        stroke = st.text_input('Stroke')

    with col2:
        physicalHealth = st.text_input('Physical Health')

    with col3:
        mentalHealth = st.text_input('Mental Health')

    with col1:
        diffWalking = st.text_input('Difference of Walking')

    with col2:
        sex = st.text_input('Sex 1->for male and 0->for female')

    with col3:
        ageCategory = st.text_input('Age')

    with col1:
        race = st.text_input('Race')

    with col2:
        diabetic = st.text_input('Diabetic 0-> for no 1-> for yes')

    with col3:
        physicalActivity = st.text_input('Physical Activity')

    with col1:
        generalHealth = st.text_input('General Health')

    with col2:
        sleepTime = st.text_input('Sleep Time')

    with col3:
        asthma = st.text_input('Asthma')

    with col1:
        kidneyDisease = st.text_input('Kidney Disease')

    with col2:
        skinCancer = st.text_input('Skin Cancer')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict(
            [[int(bmi), int(smoking), int(alcoholDrinking), int(stroke), int(physicalHealth), int(mentalHealth), int(diffWalking),
              int(sex), int(ageCategory), int(race), int(diabetic), int(
                  physicalActivity), int(generalHealth), int(sleepTime),
              int(asthma), int(kidneyDisease), int(skinCancer)]])

        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

if (selected == 'Pneumonia Prediction'):
    import streamlit as st
    import tensorflow as tf
    from PIL import Image
    import numpy as np

    # Define a function to load the trained model and make predictions on uploaded images
    def predict(image):
        # Load the trained machine learning model
        model = tf.keras.models.load_model('saved_file/pneumonia_pred.h5')

        # Preprocess the uploaded image
        img_size = (256, 256)  # Set the input size of your model
        image = image.resize(img_size)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        # Repeat the grayscale channel to match the expected input shape
        image = np.repeat(image, 3, axis=-1)

        # Make a prediction using the loaded model
        prediction = model.predict(image)

        # Convert the prediction to a human-readable format
        # ...

        # Return the prediction
        return prediction

    # Define the Streamlit app
    def app():
        # Set the page title
       # st.set_page_config(page_title='Image Classification App')

        # Set the app header
        #st.title('Image Classification App')

        # Add a file uploader to the app
        uploaded_file = st.file_uploader(
            'Upload an image', type=['jpg', 'jpeg', 'png'])

        # If an image is uploaded, display it and make a prediction
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            class_names = ['Normal', 'Infected']

            # Make a prediction using the predict() function
            prediction = predict(image)

            score = tf.nn.softmax(prediction[0])
            # Get the predicted class and confidence score
            predicted_class = class_names[np.argmax(score)]
            confidence_score = np.max(score)

            # Format the output string
            output_string = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                predicted_class, 100 * confidence_score)

            st.write(output_string)

            # Display the prediction to the user
            #st.write('hi:', prediction)

    # Run the Streamlit app
    if __name__ == '__main__':
        app()
