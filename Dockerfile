# Use an official Python runtime as a parent image
FROM python:3.10-slim

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

# NOTE: No data (.csv/.root) or trained model (.joblib/.pt) is shipped in the
# image. From a fresh clone you must first fetch data and train, e.g.:
#   docker run --rm -v ${PWD}/plots:/app/plots cern-zboson-ml \
#     sh -c "python src/data_download.py && python src/train_model.py"
# Keeping the default command as training (not realtime_simulation) avoids a
# guaranteed ModuleNotFound/FileNotFound failure on a clean checkout.
CMD ["python", "src/train_model.py"]
