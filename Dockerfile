FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

# Expose Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

# Default command (can be overridden)
CMD ["streamlit", "run", "src/app.py"]
