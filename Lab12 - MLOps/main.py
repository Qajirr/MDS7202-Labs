from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import os

# App
app = FastAPI()

# Modelo
models_path = os.path.join("models", "best_model.pkl")
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Home
@app.get("/")  
async def home():
    """
    Introducción a la API
    """
    return {
        "message": "API para predecir la potabilidad del agua",
        "input_features": [
            "ph", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic_carbon",
            "Trihalomethanes", "Turbidity"
        ],
        "output": {
            "potabilidad": "0 (No potable) o 1 (Potable)"
        }
    }
# Post para prediccion
@app.post("/predict")  
async def predict(sample: WaterSample):
    """
    Predicción de potabilidad del agua
    """
    features = np.array([[
        sample.ph,
        sample.Hardness,
        sample.Solids,
        sample.Chloramines,
        sample.Sulfate,
        sample.Conductivity,
        sample.Organic_carbon,
        sample.Trihalomethanes,
        sample.Turbidity
    ]])

    prediction = model.predict(features)[0]

    return {"potabilidad": int(prediction)}

if __name__ == "__main__":
    uvicorn.run("main:app")
