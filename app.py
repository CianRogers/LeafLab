import streamlit as st 
import tensorflow as tf
import numpy as np

# TensorFlow Model prediction function 
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image to a batch 
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('select page',['Home','About','Disease Recognition'])

# Home page 
if(app_mode == 'Home'):
    st.header('Leaf Lab')
    image_path = '/Users/cianrogers/Documents/Python/Img_Classification_Project/plant.JPEG'
    st.image(image_path,use_column_width=True)
    st.markdown(''' 
    Welcome to Leaf Lab the online Plant Disease Recognition System! 🌿🔎
    
    This application aims to effectively identify plant diseases. Upload an image of a plant leaf and the system will detect the disease.
    
    ### Get Started
    1. **Go to the Disease recognition page. Upload your image of a plant leaf with a suspected disease.
    2. **The Application will then process the image to identify disease.
    3. **View the results and recommendations for further action. 
    ''')

# About Page 
if(app_mode == 'About'):
    st.header('About')
    st.markdown('''
    This data is recreated from the offline augmentation from the original dataset. 
    The original Dataset can be found on Kaggle.                       
    ''')

elif(app_mode == 'Disease Recognition'):
    st.header('Disease Recognition')
    test_image = st.file_uploader('Choose an Image')
    if(st.button('Show Image')):
        st.image(test_image,use_column_width=True)
    # Predict button
    if(st.button('Predict')):
        with st.spinner('Please Wait...'):
            st.write('Your Prediction')
            result_index = model_prediction(test_image)
            # Define Class
            class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success('Your plant is infected with {}'.format(class_name[result_index]))
        