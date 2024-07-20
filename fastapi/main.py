from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class PredictionRequest(BaseModel):
    features: list

@app.post('/predict')
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    return {'predictions': predictions.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
