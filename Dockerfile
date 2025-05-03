# Use official Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port Heroku will bind to
EXPOSE 5000

# Start the server
CMD ["python", "app.py"]
