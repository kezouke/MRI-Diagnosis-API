# Use the official Python slim image as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file to the working directory
COPY requirements.txt . 

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "code/deployment/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
