# =========================
# Base image
# =========================
FROM python:3.11-slim

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Work directory
# =========================
WORKDIR /app

# =========================
# Install Python deps
# =========================
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Copy project
# =========================
COPY backend/ .

# =========================
# Create runtime dirs
# =========================
RUN mkdir -p uploads runs

# =========================
# Expose port
# =========================
EXPOSE 20000

# =========================
# Run server
# =========================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "20000"]
