FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/models

EXPOSE 8000

CMD ["sh", "-c", "python src/train_model.py && python -m streamlit run src/app.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true"]
