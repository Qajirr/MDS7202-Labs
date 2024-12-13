FROM python:3.11-slim

WORKDIR /app

# Requerimientos
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . /app/

EXPOSE 8000

# Iniciar App
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
