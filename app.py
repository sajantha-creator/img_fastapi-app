from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import os
import gdown

# Disable GPU usage
tf.config.set_visible_devices([], 'GPU')

# Initialize FastAPI app
app = FastAPI()

# Define the path for the model
model_path = 'image_classification_model.keras'

# Google Drive file ID for the model
file_id = '1ANXn8Bz1rpEDXJkg0TLPiQOzFeJKq9il'

# Function to download the model
def download_model():
    if not os.path.exists(model_path):
        print("Model file not found locally. Downloading from Google Drive...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)
    else:
        print("Model file found locally.")

# Download the model if necessary
download_model()

# Load the model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = {
    0: 'airplane',
    1: 'car',
    2: 'cat',
    3: 'dog',
    4: 'flower',
    5: 'fruit',
    6: 'motorbike',
    7: 'person'
}

# Define image preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize to match your model input size
    image_array = np.array(image) / 255.0  # Normalize image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define a prediction function
def predict_image(image_array: np.ndarray):
    predictions = model.predict(image_array)
    confidence_scores = predictions[0]
    predicted_class_index = int(np.argmax(confidence_scores))
    
    # Only return the top prediction
    return {
        "predicted_class_index": predicted_class_index,
        "predicted_class_name": class_labels.get(predicted_class_index, "Unknown"),
        "confidence": float(confidence_scores[predicted_class_index]),
        "top_predictions": [
            {
                "class_index": predicted_class_index,
                "confidence": float(confidence_scores[predicted_class_index]),
                "class_name": class_labels.get(predicted_class_index, "Unknown")
            }
        ]
    }

# Define the endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = preprocess_image(image)
        prediction_details = predict_image(image_array)
        return prediction_details
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
