FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "duckdb>=1.5.3" "python-dotenv>=1"
# Bake extensions into the image so startup needs no extension download.
RUN python -c "import duckdb; c=duckdb.connect(); c.execute('INSTALL quack; INSTALL httpfs')"

COPY duckdb_server.py .
CMD ["python", "duckdb_server.py"]
