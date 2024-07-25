from fastapi import FastAPI, HTTPException, Request
import uvicorn
from dotenv import load_dotenv
import os
from typing import Optional, Any

from json_manage import json_state, data_file
from model.model import build_train_model, analyse_distribute_results
from model.stock_data import fetch_stock_data, preprocess_data


load_dotenv()
model_save_name = os.getenv("MODEL_SAVE_NAME")
ticker_name = os.getenv("TICKER")
time_period = int(os.getenv("TIME_PERIOD_LSTM"))


def retrain_model():
    json_state.set_state("retraining")
    stock_data = fetch_stock_data(ticker_name)
    stock_data = preprocess_data(stock_data)
    build_train_model(stock_data, time_period=time_period, model_name=model_save_name)
    results = data_file.read(mode=2)
    analyse_distribute_results(results)
    json_state.set_state("trained")
    return


def get_nested_data(d: Any, path: str):
    keys = path.split(".")
    for key in keys:
        if "[" in key and "]" in key:
            key, index = key[:-1].split("[")
            d = d.get(key, [])
            index = int(index)
            if index >= len(d):
                raise HTTPException(status_code=404, detail="Index out of range")
            d = d[index]
        else:
            d = d.get(key)
            if d is None:
                raise HTTPException(status_code=404, detail="Path not found")
    return d


app = FastAPI()


@app.get("/")
async def get(query: Optional[str] = None):
    json = json_state.get_state()
    print(f"get, json_state {json}")
    if query:
        try:
            result = get_nested_data(json, query)
            return result
        except HTTPException as e:
            return {"error": e.detail}
    return json


@app.get("/data/")
async def get(query: Optional[str] = None):
    json = data_file.read()
    print(f"get, data {json}")
    if query:
        try:
            result = get_nested_data(json, query)
            return result
        except HTTPException as e:
            return {"error": e.detail}
    return json


@app.post("/")
async def retrain(request: Request):
    try:
        # Parse the incoming JSON data
        data = await request.json()

        # Check if the key 'status' with value 'retrain' exists
        if data.get("status") == "retrain":
            print("Retrain model")
            retrain_model()
            return {"message": "Model retrained."}
        else:
            raise HTTPException(status_code=400, detail="Invalid status value")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
