FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TERM=xterm-256color

WORKDIR /app

# Install dependencies first for better Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime files to keep the image small.
COPY agent ./agent
COPY agent_build ./agent_build
COPY run_tui.py quickstart.py ./


CMD ["python", "run_tui.py"]
