import fastapi
from pydantic import BaseModel
import uvicorn

async def train_model():
    print("ASYNC training model.")
    return {"1d": 0, "7d": 1}

app = fastapi.FastAPI()

class PredictionRequest(BaseModel):
    features: list


@app.post('/')
async def predict(request: PredictionRequest):
    prediction = await train_model()
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)




