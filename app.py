import streamlit as st
import pickle
import numpy as np
import time

# Load the model and scaler
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Mapping of prediction to class names
class_names = {0: 'Benign', 1: 'Malignant'}

# Title of the app
st.title('Breast Cancer Prediction')

st.write("Please enter the values for the following features in the sidebar:")

# Input fields in the sidebar with instructions
with st.sidebar.form(key='breast_cancer_form'):
    st.write("### Enter the details for each feature below:")
    st.write("Ensure the values are within typical ranges for accurate predictions.")

    radius_mean = st.number_input('Radius Mean (e.g., 14.0)', min_value=0.0, value=0.0, help="Average radius of the cell nuclei")
    texture_mean = st.number_input('Texture Mean (e.g., 19.0)', min_value=0.0, value=0.0, help="Standard deviation of gray-scale values")
    perimeter_mean = st.number_input('Perimeter Mean (e.g., 90.0)', min_value=0.0, value=0.0, help="Average perimeter of the cell nuclei")
    area_mean = st.number_input('Area Mean (e.g., 600.0)', min_value=0.0, value=0.0, help="Average area of the cell nuclei")
    smoothness_mean = st.number_input('Smoothness Mean (e.g., 0.1)', min_value=0.0, value=0.0, help="Mean value of local variation in radius lengths")
    compactness_mean = st.number_input('Compactness Mean (e.g., 0.2)', min_value=0.0, value=0.0, help="Mean of (perimeter^2 / area - 1.0)")
    concavity_mean = st.number_input('Concavity Mean (e.g., 0.3)', min_value=0.0, value=0.0, help="Mean of severity of concave portions of the contour")
    concave_points_mean = st.number_input('Concave Points Mean (e.g., 0.15)', min_value=0.0, value=0.0, help="Mean number of concave portions of the contour")
    symmetry_mean = st.number_input('Symmetry Mean (e.g., 0.2)', min_value=0.0, value=0.0, help="Mean of symmetry value")
    fractal_dimension_mean = st.number_input('Fractal Dimension Mean (e.g., 0.06)', min_value=0.0, value=0.0, help="Mean value of fractal dimension")
    
    radius_se = st.number_input('Radius SE (e.g., 0.5)', min_value=0.0, value=0.0, help="Standard error for radius")
    texture_se = st.number_input('Texture SE (e.g., 1.2)', min_value=0.0, value=0.0, help="Standard error for texture")
    perimeter_se = st.number_input('Perimeter SE (e.g., 2.5)', min_value=0.0, value=0.0, help="Standard error for perimeter")
    area_se = st.number_input('Area SE (e.g., 50.0)', min_value=0.0, value=0.0, help="Standard error for area")
    smoothness_se = st.number_input('Smoothness SE (e.g., 0.01)', min_value=0.0, value=0.0, help="Standard error for smoothness")
    compactness_se = st.number_input('Compactness SE (e.g., 0.03)', min_value=0.0, value=0.0, help="Standard error for compactness")
    concavity_se = st.number_input('Concavity SE (e.g., 0.04)', min_value=0.0, value=0.0, help="Standard error for concavity")
    concave_points_se = st.number_input('Concave Points SE (e.g., 0.02)', min_value=0.0, value=0.0, help="Standard error for concave points")
    symmetry_se = st.number_input('Symmetry SE (e.g., 0.02)', min_value=0.0, value=0.0, help="Standard error for symmetry")
    fractal_dimension_se = st.number_input('Fractal Dimension SE (e.g., 0.003)', min_value=0.0, value=0.0, help="Standard error for fractal dimension")

    radius_worst = st.number_input('Radius Worst (e.g., 25.0)', min_value=0.0, value=0.0, help="Worst radius of the cell nuclei")
    texture_worst = st.number_input('Texture Worst (e.g., 35.0)', min_value=0.0, value=0.0, help="Worst texture of the cell nuclei")
    perimeter_worst = st.number_input('Perimeter Worst (e.g., 160.0)', min_value=0.0, value=0.0, help="Worst perimeter of the cell nuclei")
    area_worst = st.number_input('Area Worst (e.g., 2000.0)', min_value=0.0, value=0.0, help="Worst area of the cell nuclei")
    smoothness_worst = st.number_input('Smoothness Worst (e.g., 0.2)', min_value=0.0, value=0.0, help="Worst smoothness value")
    compactness_worst = st.number_input('Compactness Worst (e.g., 0.6)', min_value=0.0, value=0.0, help="Worst compactness value")
    concavity_worst = st.number_input('Concavity Worst (e.g., 0.7)', min_value=0.0, value=0.0, help="Worst concavity value")
    concave_points_worst = st.number_input('Concave Points Worst (e.g., 0.2)', min_value=0.0, value=0.0, help="Worst number of concave points")
    symmetry_worst = st.number_input('Symmetry Worst (e.g., 0.3)', min_value=0.0, value=0.0, help="Worst symmetry value")
    fractal_dimension_worst = st.number_input('Fractal Dimension Worst (e.g., 0.1)', min_value=0.0, value=0.0, help="Worst fractal dimension value")

    submit_button = st.form_submit_button('Predict')

# Prediction button click event
if submit_button:
    try:
        # Create a list of all features
        features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                    concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, 
                    texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, 
                    concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, 
                    perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, 
                    concave_points_worst, symmetry_worst, fractal_dimension_worst]

        # Convert to numpy array
        features = np.array([features])

        # Simulate a loading time of 2 seconds
        with st.spinner('Predicting...'):
            time.sleep(2)

        # Scale the features
        scaled_features = scaler.transform(features)

        # Perform prediction
        prediction = model.predict(scaled_features)

        # Map the prediction to class name
        predicted_class = class_names[prediction[0]]

        # Show the prediction result
        st.success(f"The predicted class is: {predicted_class}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
