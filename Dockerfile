FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# If requirements.txt doesn't exist yet, we'll install from setup.py or manually.
# For now, let's assume we install via pip install .

COPY . .

RUN pip install --no-cache-dir -e .

# Expose Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

# Default command (can be overridden)
CMD ["streamlit", "run", "src/app.py"]
