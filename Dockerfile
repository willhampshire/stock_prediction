# Use the FastAPI base image with a suitable Python version
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

LABEL authors="williamhampshire"

# Copy FastAPI requirements
COPY requirements.txt .

# Install FastAPI dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy code to working dir
COPY . .

# Expose the port the FastAPI app runs on
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
