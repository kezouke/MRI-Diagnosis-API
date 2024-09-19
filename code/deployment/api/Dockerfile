# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable for the Uvicorn command
ENV UVICORN_CMD="uvicorn code.deployment.api.api:app --reload"

# Expose the port that Uvicorn will run on
EXPOSE 8000

CMD ["uvicorn", "code.deployment.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
