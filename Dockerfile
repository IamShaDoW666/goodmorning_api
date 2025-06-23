# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 6666

# Run the FastAPI server
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "6666"]
