FROM ubuntu:latest
LABEL authors="williamhampshire"

ENTRYPOINT ["top", "-b"]
# Base image for FastAPI
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set the working directory
WORKDIR /app

# Copy FastAPI requirements
COPY ./app/requirements.txt /app/requirements.txt

# Install FastAPI dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy FastAPI application
COPY ./app /app

# Copy the pre-trained model and scaler
COPY ./model.pkl /app/model.pkl
COPY ./scaler.pkl /app/scaler.pkl

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
