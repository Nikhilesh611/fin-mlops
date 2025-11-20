# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1
# Add non-essential packages that will be used for API and monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install all dependencies (including API ones)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script
COPY api.py /app/

# The training script (train.py) is also copied, but the ENTRYPOINT is changed

# Expose the port for the API
EXPOSE 8000 

# Command to run the Uvicorn server (production serving)
# The --reload flag is only for development, use uvicorn with worker for production
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]