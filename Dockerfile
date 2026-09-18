# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Windows hosts default to cp1252, which crashed emoji prints; force UTF-8
# everywhere so container logs match local runs.
ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Create a volume for plots so they can be accessed outside the container
VOLUME ["/app/plots"]

# Default command downloads data (if missing) and trains the model.
CMD ["sh", "-c", "python src/data_download.py && python src/train_model.py"]
