# Stock market prediction

## Setup
* Clone repository
* Run `pip install -r requirements.txt` in main project directory (or use conda environment - kept getting segmentation fault)

### Docker setup (optional)
* Install docker from [docker.com](docker.com)
* Run `docker build -t stock_predict_containter .` in project directory to create docker container


## Usage
Run `start.py`, and enter company tag.
See status, data and results using FastAPI and streamlit.
The model will start training if it has not already. Hosting will start promptly on localhost ports 8000 and 8501. 
The model has 2 output nodes - prediction of +1d, and +7d.

### FastAPI Endpoints
Accessible on port 8000. Rested behaviour.

`localhost:8000/` : root shows status. Query with ?query=status to get status as string from json.

`localhost:8000/data` : output data comparing real data vs. predicted data. Split into [1d, 7d] lists. Query with ?query= to index the results json. Can also be sliced, e.g. `/data?query=real[0]`

To retrain model, `POST` to `/retrain` with `{status: "retrain"}` - await response `200`, with json response `{"message": "Model retrained."}`
Status on root endpoint will reflect.

### Streamlit
Accessible on port 8501.

Depicts two line plots, to indicate the real vs. predicted values produced by the model. This is to analyse the accuracy,
and idendify areas for improvement. For example, the model might be a delayed curve of the real data, and may poorly 
reflect the true volatility.

Streamlit allows for an easy, clean way of hosting the data on localhost.

## Fixing things
Sometimes, the process running on localhost:8000 will not quit properly, meaning a new process cannot start.
On macos, run `lsof -i :8000`, and then `kill -9 <PID>` to kill processes that aren't still supposed to be running.

I had a lot of problems getting packages to work together on python 3.11, and ended up using a conda environment. Perhaps Python 3.10 is more inter-compatible with all packages, or perhaps one library had a bug affecting all.

To get docker working, I had to open `settings.json` in /Library/Group\ Containers/group.com.docker/settings.json, add `"kernelForUDP": false`, delete the contents of the filesharingDirectories key (e.g. leave empty list), and restart computer.



## Frameworks

- `docker` (v20.0.6 for macos 11.6, support discontinued)
- `fastapi`

## Code practices
Used the following for good code practice and safety:
- `pydantic`
- `mypy`
- `black`

