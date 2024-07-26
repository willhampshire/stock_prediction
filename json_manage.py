import json
from typing import List, Tuple, Any
import numpy as np
from pydantic import BaseModel
import time

STATE_FILE = "json/state.json"
DATA_FILE = "json/data.json"


class StatusJSON(BaseModel):
    status: str


class MessageResponse(BaseModel):
    message: str


class DataType(BaseModel):
    real: List[List[float]] | np.ndarray
    prediction: List[List[float]] | np.ndarray


class ArrayException(Exception):
    """Exception raised for errors in the input array sizes."""

    def __init__(self, message="Array size is incorrect"):
        self.message = message
        super().__init__(self.message)


class StateMachine:
    def __init__(self, init_state: str = str(None)):
        self.json: StatusJSON

        self._load_state()

        if init_state != None:
            self.set_state(str(init_state))
            self._load_state()

    def _load_state(self) -> bool:
        try:
            with open(STATE_FILE, "r") as file:
                self.json = json.load(file)
                return True
        except (FileNotFoundError, json.JSONDecodeError):
            self.json = StatusJSON(status="unknown")
            return False

    def _save_state(self) -> bool:
        with open(STATE_FILE, "w") as file:
            json.dump(self.json, file)
            return True

    def get_state(self) -> StatusJSON:
        self._load_state()
        return self.json

    def set_state(self, value: str) -> bool:
        self.json = StatusJSON(status=value)
        self._save_state()
        return True


class Data:
    def __init__(self):
        self._cache = self.read(mode=1)
        pass

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
            with open(DATA_FILE, "w") as file:
                json.dump(data, file)
                return True
        except:
            data = DataType(real=np.empty((2)), prediction=np.empty((2)))
            with open(DATA_FILE, "w") as file:
                json.dump(data, file)
                return False

    def read(
        self, mode=1
    ) -> Tuple[List[List], List[List]] | DataType | MessageResponse:
        """
        Read the real and prediction 2D arrays from a JSON file.

        Returns:
            Tuple[List[List[float], List[float]], List[List[float], List[float]]]
            or DataType
            or MessageResponse:
            Tuple containing the real and prediction arrays.
        """
        try:
            with open(DATA_FILE, "r") as file:
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
