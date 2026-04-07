FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY my_env ./my_env
COPY config.py .
COPY inference.py .
COPY openai_client.py .
COPY openenv.yaml .
COPY README.md .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
