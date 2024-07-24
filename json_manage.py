import json
from typing import List, Tuple, Any

STATE_FILE = 'json/state.json'
DATA_FILE = 'json/data.json'


class StateMachine:
    def __init__(self):
        self.load_state()

    def load_state(self):
        try:
            with open(STATE_FILE, 'r') as file:
                self.json = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.json = {"status": "unknown"}

    def save_state(self):
        with open(STATE_FILE, 'w') as file:
            json.dump(self.json, file)

    def get_state(self, key: str):
        return self.json.get(key, "Key not found")

    def set_state(self, value: Any, key: str = "status"):
        self.json[key] = value
        self.save_state()


class Data:
    def __init__(self):
        self.real = []
        self.prediction = []

    def write(self, real: List[List[float]], prediction: List[List[float]]):
        """
        Write the real and prediction 2D arrays to a JSON file.

        Args:
            real (List[List[float]]): 2D array of real values.
            prediction (List[List[float]]): 2D array of prediction values.
        """
        data = {
            "real": real,
            "prediction": prediction
        }
        with open(DATA_FILE, 'w') as file:
            json.dump(data, file)

    def read(self) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Read the real and prediction 2D arrays from a JSON file.

        Returns:
            Tuple[List[List[float]], List[List[float]]]: Tuple containing the real and prediction 2D arrays.
        """
        try:
            with open(DATA_FILE, 'r') as file:
                data = json.load(file)
                real = data.get("real", [])
                prediction = data.get("prediction", [])
                return (real, prediction)
        except (FileNotFoundError, json.JSONDecodeError):
            return -1


json_state = StateMachine()
data_file = Data()
