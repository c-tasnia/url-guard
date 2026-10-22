FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY urlguard ./urlguard
COPY app.py ./
COPY static ./static
COPY model ./model

# Hugging Face Spaces runs containers as a non-root user (uid 1000).
RUN useradd -m -u 1000 appuser
USER appuser

# 7860 is the Hugging Face Spaces default. Hosts that set PORT (Render, Railway)
# override it.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
