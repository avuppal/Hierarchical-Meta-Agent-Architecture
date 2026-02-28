# Dockerfile for HMA Research Environment
# Use a stable base image with Python and necessary libraries (like CUDA base if we run local models later)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages (git is needed for the final push)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file (though we will install directly for now)
COPY requirements.txt . 

# Install Python dependencies. 
# NOTE: Since pip/pip3 failed earlier, we will rely on the sub-agent/you to run these commands *after* container build if necessary, 
# but we include the file for completeness of the structure.
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the existing project code into the container
COPY agent_budget_research /app/agent_budget_research

# The script we want to run to start the simulation
CMD ["python3", "agent_budget_research/hma_orchestrator.py"]