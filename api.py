# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Define the FastAPI application
app = FastAPI(title="Financial Sentiment API")

# Define the input data structure
class TextIn(BaseModel):
    text: str

# Define the output data structure
class PredictionOut(BaseModel):
    sentiment: str
    confidence: float

# Global variables for model/tokenizer
tokenizer = None
model = None
LABEL_MAP = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}
MODEL_ARTIFACTS_PATH = "./model_artifacts" # This path is where Jenkins archives the model

@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer only once when the server starts."""
    global tokenizer, model

    # Ensure the model directory exists
    if not os.path.exists(MODEL_ARTIFACTS_PATH):
        raise FileNotFoundError(f"Model artifacts not found at {MODEL_ARTIFACTS_PATH}. Check CI/CD pipeline.")

    print(f"Loading model from {MODEL_ARTIFACTS_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ARTIFACTS_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ARTIFACTS_PATH)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real environment, you might stop the service here

@app.post("/predict", response_model=PredictionOut)
async def predict_sentiment(text_in: TextIn):
    """Perform sentiment prediction on the input text."""
    # Tokenize the input text
    inputs = tokenizer(text_in.text, return_tensors="pt", truncation=True, padding=True)

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Process the output (logits)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_class_id = torch.argmax(probabilities).item()

    sentiment = LABEL_MAP[predicted_class_id]
    confidence = probabilities[predicted_class_id].item()

    return PredictionOut(sentiment=sentiment, confidence=confidence)

# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}