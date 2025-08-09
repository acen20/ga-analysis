# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt first (for layer caching)
COPY ./src/requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Copy the rest of your application code
COPY ./src .

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]