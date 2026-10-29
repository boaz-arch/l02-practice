# Stage 1: install dependencies and train the model
FROM python:3.12-slim AS build
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .
RUN python train.py

# Stage 2: a small runtime image that runs as a non-root user
FROM python:3.12-slim
WORKDIR /app
COPY --from=build /venv /venv
COPY --from=build /app/model.joblib .
COPY app.py .
ENV PATH="/venv/bin:$PATH"
USER 1000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
