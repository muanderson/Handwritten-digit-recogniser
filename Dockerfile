# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Expose the port the app runs on (Hugging Face default)
EXPOSE 7860

# Define the command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]