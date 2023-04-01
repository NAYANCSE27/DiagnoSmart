import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define a function to load the trained model and make predictions on uploaded images
def predict(image):
    # Load the trained machine learning model
    model = tf.keras.models.load_model('my_model.h5')
    
    # Preprocess the uploaded image
    img_size = (256, 256) # Set the input size of your model
    image = image.resize(img_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) # Add batch dimension
    image = np.expand_dims(image, axis=-1) # Add channel dimension
    image = np.repeat(image, 3, axis=-1) # Repeat the grayscale channel to match the expected input shape
    
    # Make a prediction using the loaded model
    prediction = model.predict(image)
    
    # Convert the prediction to a human-readable format
    # ...
    
    # Return the prediction
    return prediction

# Define the Streamlit app
def app():
    # Set the page title
    st.set_page_config(page_title='Image Classification App')
    
    # Set the app header
    st.title('Image Classification App')
    
    # Add a file uploader to the app
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    
    # If an image is uploaded, display it and make a prediction
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        class_names=['Normal' ,'Infected']
        
        # Make a prediction using the predict() function
        prediction = predict(image)
        
        score = tf.nn.softmax(prediction[0])
        # Get the predicted class and confidence score
        predicted_class = class_names[np.argmax(score)]
        confidence_score = np.max(score)

        # Format the output string
        output_string = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(predicted_class, 100 * confidence_score)
        
        st.write(output_string)
        
        # Display the prediction to the user
        #st.write('hi:', prediction)
        

# Run the Streamlit app
if __name__ == '__main__':
    app()
