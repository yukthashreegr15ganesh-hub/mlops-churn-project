FROM python:3.10-slim

WORKDIR /app

# Copy root requirements
COPY requirements.txt .

# Install dependencies (fix numpy + mlflow)
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
