# Use official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Ensure output is sent straight to terminal (no buffering)
ENV PYTHONUNBUFFERED=1

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Start the application with Uvicorn, binding to all interfaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]