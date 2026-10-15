FROM python:3.12-slim
WORKDIR /app
# EXTRAS=true adds Qdrant, LangChain and PDF support (requirements-extras.txt).
ARG EXTRAS=false
COPY requirements.txt requirements-extras.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$EXTRAS" = "true" ]; then pip install --no-cache-dir -r requirements-extras.txt; fi
COPY app/ app/
COPY data/ data/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
