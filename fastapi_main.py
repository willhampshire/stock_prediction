from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Optional, Any
import configparser

from json_manage import (
    json_state,
    data_file,
    metrics_file,
    StatusJSON,
    MessageResponse,
    DataType,
)
from model.model import build_train_model, analyse_distribute_results
from model.stock_data import fetch_stock_data, preprocess_data


config = configparser.ConfigParser()
config.read("config.ini")

model_save_name = config["MODEL"]["MODEL_SAVE_NAME"]
ticker_name = config["STOCKDATA"]["TICKER"]
time_period = int(config["STOCKDATA"]["TIME_PERIOD_LSTM"])


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
    # print(f"get, json_state {json}")
    if query:
        try:
            result = get_nested_data(json, query)
            return result
        except HTTPException as e:
            return MessageResponse(message=e.detail)
    return json


@app.get("/data/")
async def get(query: Optional[str] = None):
    json: DataType = data_file.read()
    # print(f"get, data {json}")
    if query:
        try:
            result = get_nested_data(json, query)
            return result
        except HTTPException as e:
            return MessageResponse(message=e.detail)
    return json


@app.get("/metrics/")
async def get(query: Optional[str] = None):
    json: DataType = metrics_file.read()
    # print(f"get, data {json}")
    if query:
        try:
            result = get_nested_data(json, query)
            return result
        except HTTPException as e:
            return MessageResponse(message=e.detail)
    return json


@app.post("/", response_model=MessageResponse)
async def retrain(retrain_request: StatusJSON):
    if retrain_request.status == "retrain":
        print("Retrain model")
        retrain_model()
        return MessageResponse(message="Model retrained.")
    else:
        raise HTTPException(status_code=400, detail="Invalid status value")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
