# Use a small Python base image
FROM python:3.10-slim

# Set a working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 8000 (or whichever your app uses)
EXPOSE 5000

# Command to run your app — adjust as needed!
# Here assuming you run app.py with uvicorn, a common choice for FastAPI apps
CMD ["python", "app.py"]
