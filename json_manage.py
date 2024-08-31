import pandas as pd

import json
from typing import List, Tuple
import numpy as np
from pydantic import BaseModel
from pathlib import Path


STATE_FILE = "json/state.json"
DATA_FILE = "json/data.json"
METRICS_FILE = "json/metrics.json"


class StatusJSON(BaseModel):
    status: str


class MessageResponse(BaseModel):
    message: str


class DataType(BaseModel):
    real: List[List[float]] | np.ndarray
    prediction: List[List[float]] | np.ndarray

    class Config:
        arbitrary_types_allowed = True


class ArrayException(Exception):
    """Exception raised for errors in the input array sizes."""

    def __init__(self, message="Array size is incorrect"):
        self.message = message
        super().__init__(self.message)


class StateMachine:
    def __init__(self):
        self.json: StatusJSON

        self.state_file_path = Path(STATE_FILE)

        if self.state_file_path.exists():
            self._load_state()
            print(f"{STATE_FILE} already exists")
        else:
            self.json = StatusJSON(status="created json")
            self._save_state()

    def _load_state(self) -> bool:
        try:
            with open(self.state_file_path, "r") as file:
                self.json = json.load(file)
                return True
        except (FileNotFoundError, json.JSONDecodeError):
            self.json = StatusJSON(status="unknown")
            return False

    def _save_state(self) -> bool:
        with open(self.state_file_path, "w") as file:
            print(f"Dumping JSON val - {self.json.dict()}")
            json.dump(self.json.dict(), file)
            return True

    def get_state(self) -> StatusJSON:
        self._load_state()
        return self.json

    def set_state(self, value: str) -> bool:
        self.json = StatusJSON(status=value)
        self._save_state()
        return True


class Metrics:
    def __init__(self):
        self.data_file_path = Path(METRICS_FILE)

        if self.data_file_path.exists():
            print(f"{METRICS_FILE} already exists")
        else:
            self.write({"metrics": "no metrics yet"})

        self._cache = self.read()

    def write(self, metrics: dict) -> bool:
        """
        Writes metrics to JSON file
        """
        blank = {"error": "could not write metrics"}
        try:
            with open(self.data_file_path, "w") as file:
                json.dump(metrics, file)
                return True
        except:
            with open(self.data_file_path, "w") as file:
                json.dump(blank, file)
                return False

    def read(self) -> dict | MessageResponse:
        """
        Read the metrics from a JSON file.

        Returns:
            dict
            or MessageResponse
        """
        try:
            with open(self.data_file_path, "r") as file:
                data = json.load(file)

                return data

        except (FileNotFoundError, json.JSONDecodeError):
            return MessageResponse(message="Data not found")


class Data:
    def __init__(self):
        self.data_file_path = Path(DATA_FILE)

        if self.data_file_path.exists():
            print(f"{DATA_FILE} already exists")
        else:
            self.write([[]], [[]])

        self._cache = self.read(mode=1)

    def write(
        self,
        real: List[List[float]] | np.ndarray,
        prediction: List[List[float]] | np.ndarray,
    ) -> bool:
        """
        Write the real and prediction 2D arrays to a JSON file.

        Args:
            real (List[List[float]]): 2D array of real values. Can be np.ndarray.
            prediction (List[List[float]]): 2D array of prediction values. Can be np.ndarray.
        """

        if len(np.array(real).shape) > 2 or len(np.array(prediction).shape) > 2:
            raise ArrayException

        # convert to standard python types
        real_list = np.array(real).astype(float).tolist()
        prediction_list = np.array(prediction).astype(float).tolist()

        data: DataType = DataType(real=real_list, prediction=prediction_list)
        try:
            with open(self.data_file_path, "w") as file:
                json.dump(data.dict(), file)
                return True
        except:
            data = DataType(real=np.empty((2)), prediction=np.empty((2)))
            with open(self.data_file_path, "w") as file:
                json.dump(data.dict(), file)
                return False

    def read(
        self, mode=1
    ) -> Tuple[List[List], List[List]] | DataType | MessageResponse:
        """
        Read the real and prediction 2D arrays from a JSON file.

        Returns:
            Tuple[List[List[float], List[float]], List[List[float], List[float]]]
            or DataType
            or MessageResponse

            when mode=2:
            Tuple containing the real and prediction arrays.
        """
        try:
            with open(self.data_file_path, "r") as file:
                data = json.load(file)
                if mode == 1:
                    return data
                if mode == 2:
                    real = data.get("real", None)
                    prediction = data.get("prediction", None)
                    return (real, prediction)
        except (FileNotFoundError, json.JSONDecodeError):
            return MessageResponse(message="Data not found")


json_state = StateMachine()
data_file = Data()
metrics_file = Metrics()


if __name__ == "__main__":  # test json state functionality
    json_state.set_state("testing...")
    print(json_state.get_state())
