# Stock market prediction

## Setup
* Install docker from [docker.com](docker.com)
* Clone repository
* Run `docker-compose up --build` in project `venv` terminal
* Run `pip install -r requirements.txt` in main project directory (or use conda environment - kept getting segmentation fault)

## Usage
Run `start.py`, and enter company tag.
See results by ____.

## Fixing things
Sometimes, the process running on localhost:8000 will not quit properly, meaning a new process cannot start.
On macos, run `lsof -i :8000`, and then `kill -9 <PID>` to kill processes that aren't still supposed to be running.

I had a lot of problems getting packages to work together on python 3.11, and ended up using a conda environment.

## Results
`streamlit`


## Frameworks

- `docker` (v20.0.6 for macos 11.6, support discontinued)
- `fastapi`


## Code practices
Used the following for good code practice and safety:
- `pydantic`
- `mypy`
- `black`

