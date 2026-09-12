FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests

COPY deployment/frontend /app/deployment/frontend

EXPOSE 8501
ENV API_BASE_URL=http://api:8000

CMD ["streamlit", "run", "deployment/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
