# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt first (for layer caching)
COPY ./src/requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of your application code
COPY ./src .

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]