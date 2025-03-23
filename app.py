from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib

# Define your trained model architecture
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and scaler
model = RegressionModel(input_dim=8)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load("scaler.pkl")

# FastAPI app
app = FastAPI()

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                      features.AveBedrms, features.Population, features.AveOccup,
                      features.Latitude, features.Longitude]])

    scaled_data = scaler.transform(data)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    return {"predicted_price": round(prediction, 2)}