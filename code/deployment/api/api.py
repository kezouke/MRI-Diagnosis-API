from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import torch
import io
from PIL import Image
from torchvision import transforms
import os
from code.models.model_a import BrainTumorClassifier, get_project_root
from code.utils.get_data_loaders import get_project_root, create_data_loaders

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str


class ErrorResponse(BaseModel):
    error: str


class AccuracyResponse(BaseModel):
    test_accuracy: float


class PredictionRequest(BaseModel):
    file: str


# Initialize FastAPI app
app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.
                      backends.mps.is_available() else "cpu")
num_classes = 4
model = BrainTumorClassifier(num_classes=num_classes).to(device)
model_path = os.path.join(get_project_root(), 'models',
                          'model_a.pth')  # Use your best model file
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/predict/",
          response_model=PredictionResponse,
          responses={500: {
              "model": ErrorResponse
          }},
          summary="Predict brain tumor type",
          description=
          "Upload an image of an MRI scan to predict the type of brain tumor.")
async def predict(file: UploadFile = File(...)):
    """
    Upload an MRI scan image to get a prediction on the type of brain tumor.

    Example:
    ```
    curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@/path/to/your/image.jpg"
    ```
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform the image
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        class_names = [
            'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'
        ]
        predicted_label = class_names[predicted_class]
        return JSONResponse(content={"prediction": predicted_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get(
    "/evaluate/",
    response_model=AccuracyResponse,
    responses={500: {
        "model": ErrorResponse
    }},
    summary="Evaluate model accuracy",
    description="Evaluate the model's accuracy using the test data loader.")
async def evaluate():
    """
    Evaluate the accuracy of the model using the test dataset.

    Example:
    ```
    curl -X GET "http://127.0.0.1:8000/evaluate/"
    ```
    """
    try:
        # Create the data loaders
        _, _, test_loader = create_data_loaders()

        # Evaluate model accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return JSONResponse(content={"test_accuracy": accuracy})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# To run the server, use the command below in the terminal:
# uvicorn api:app --reload
