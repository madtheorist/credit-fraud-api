FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

COPY main.py /app/main.py

# Expose port used by FastAPI
EXPOSE 8000

COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]