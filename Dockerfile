FROM python:3.9.13-slim


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE 4096

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY demo.py /demo.py
COPY app/ /app/

CMD ["streamlit", "run", "/demo.py"]
