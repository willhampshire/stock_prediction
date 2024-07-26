# Use the FastAPI base image with a suitable Python version
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

LABEL authors="williamhampshire"

# Copy FastAPI requirements
COPY requirements.txt .

# Install FastAPI dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy code to working dir
COPY . .

# Expose the ports
EXPOSE 8000 8501

# Make executable
RUN chmod +x /start.py

# Run start.py
CMD ["python", "/start.py"]

