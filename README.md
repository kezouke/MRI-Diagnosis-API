# Brain MRI Tumor Detection and Classification

## Abstract

Brain tumors are a significant health concern, accounting for 85-90% of all primary Central Nervous System (CNS) tumors. Annually, around 11,700 individuals are diagnosed with brain tumors, with a 5-year survival rate of approximately 34% for men and 36% for women. Brain tumors can be benign, malignant, pituitary tumors, etc. Early and accurate diagnosis is crucial for effective treatment and improving patient outcomes. Magnetic Resonance Imaging (MRI) is the most effective technique for detecting brain tumors, generating a substantial amount of image data that is typically reviewed by radiologists. However, manual examination can be prone to errors due to the complexity of brain tumors.

Automated classification techniques using Machine Learning (ML) and Artificial Intelligence (AI) have shown higher accuracy compared to manual methods. This project proposes a system that leverages Deep Learning algorithms, including Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN), and Transfer Learning (TL), to enhance the detection and classification of brain tumors, aiding doctors worldwide.

## Context

Brain tumors present significant challenges due to their complexity and variability in size and location. Accurate analysis of MRI scans often requires specialized neurosurgeons, and in developing countries, the shortage of skilled professionals makes it challenging to generate timely and accurate reports. An automated cloud-based system can alleviate these challenges by providing consistent and efficient analysis of MRI scans.

## Definition

This project aims to detect and classify brain tumors using CNN techniques as part of a Deep Learning approach.

### Main Task

The main task of this repository involves deploying a machine learning model as an API and creating a web application that interacts with this API. The model API will handle requests from the web application and provide responses. The web application should include input fields for uploading MRI images, a button to trigger predictions, and an area to display the results. The deployment will use Docker containers, FastAPI for the model API, and Streamlit for the web application.

