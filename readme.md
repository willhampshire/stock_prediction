# Stock market prediction
A project to practice making a local machine learning pipeline. Uses LSTM network, FastAPI, and streamlit, to train, manage, predict and display stock market 
predictions for a single company.

## To Do List
- Predict a moving average instead of daily results, as model does not predict volatility

## Summary
Although not an accurate model, I learned a lot about model selection and where models can break down, and not be useful.
I also learned how to improve code and file structure to work well with FastAPI and Streamlit.
For example, it has an extremely poor `R^2` result, mainly due to the poor prediction of random volatility. 

It is interesting to observe that the 1d and 7d prediction is very similar in form, which could indicate some viability.

I plan to continue to play with different neural network structures to see if anything interesting happens, 
and to see if any of the known issues with LSTM stock prediction are reduced.

## Setup
* Clone repository
* Run `pip install -r requirements_converted.txt` in main project directory (or use conda environment - kept getting segmentation fault)

### Docker setup (optional)
* Install docker from [docker.com](https://www.docker.com/)
* Run `docker build -t stock_predict_containter .` in project directory to create docker container


## Usage
Configure `config.ini`, and run `start.py`.
See status, data and results using FastAPI and streamlit.
The model will start training if it has not already. Hosting will start promptly on localhost ports 8000 and 8501. 
The model has 2 output nodes - prediction of +1d, and +7d.

### Config
`MODEL_SAVE_NAME`: saves the model to working directory. Use .keras or .h5.

`TICKER`: company ticker to obtain the stocks data.

`TIME_PERIOD_LSTM`: window size in days (sample rate of yfinance) to create windows to train the LSTM layer specifically.

### FastAPI Endpoints
Accessible on port 8000. 

Rested behaviour. For example,
query with ?query= to index the results json. 
Can also be sliced, e.g. `/data?query=real[0]`, and nested data returned using `.` to separate keys, e.g. `/metrics?query=1d.MSE`


`localhost:8000/` : root shows status. Query with ?query=status to get status as string from json.

`localhost:8000/data` : output data comparing real data vs. predicted data. Split into [1d, 7d] lists, [0, 1] respectively. 
Not addressable as keys as is a list.

`localhost:8000/metrics` : model evaluation metrics MSE, MAE, R^2

To retrain model, `POST` to `/retrain` with `{status: "retrain"}` - await response `200`, with json response `{"message": "Model retrained."}`
Status on root endpoint will reflect.

### Streamlit
Accessible on port 8501. Does not render in Safari.

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

- `docker` (v24.0.6 for macos 11.6)
- `fastapi`
- `streamlit`

## Code formatting
Used the following for formatting and type safety:
- `pydantic`
- `typing` standard library
- `mypy` - used to check code for type ambiguity and mismatch
- `black` - formats Python code to PEP 8 on save


