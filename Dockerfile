FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY pvsc ./pvsc
COPY data ./data
RUN pip install --no-cache-dir .
ENTRYPOINT ["python", "-m", "pvsc"]
