import streamlit as st
import requests
from PIL import Image
import io

# Set the title of the web app
st.title("Brain Tumor Classification")

# Add a file uploader for MRI scan images
uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "jpeg", "png"])

# API URL (assuming the FastAPI server is running locally)
api_url = "http://fastapi-service:8000/predict/"

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI scan", use_column_width=True)

    # Prepare the image for the API request
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Create a button to submit the image for prediction
    if st.button("Predict Tumor Type"):
        with st.spinner('Sending image for prediction...'):
            # Send the image to the FastAPI model server for prediction
            try:
                files = {"file": (uploaded_file.name, img_byte_arr, "image/jpeg")}
                response = requests.post(api_url, files=files)

                if response.status_code == 200:
                    prediction = response.json().get("prediction")
                    st.success(f"The model predicts: {prediction}")
                else:
                    st.error("Error in prediction: " + response.json().get("error"))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
