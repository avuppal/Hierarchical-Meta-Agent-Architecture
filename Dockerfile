# Dockerfile for HMA Research Environment (AaaS Mode)
# Use a stable base image with Python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt . 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the existing project code into the container
COPY . /app/agent_budget_research

# Default environment variables for vLLM (can be overridden at runtime)
ENV VLLM_API_BASE="http://host.docker.internal:8000/v1"
ENV HMA_API_BASE="http://host.docker.internal:8000/v1"

# The command to run the FastAPI service via Uvicorn
CMD ["uvicorn", "agent_budget_research.service:app", "--host", "0.0.0.0", "--port", "8080"]
