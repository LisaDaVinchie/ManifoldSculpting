FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/run.py ./src/
COPY src/ManifoldSculpting.py ./src/
COPY src/dataset_generation.py ./src/

CMD ["python3", "-u", "src/run.py"]