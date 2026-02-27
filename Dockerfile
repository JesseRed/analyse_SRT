FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local dependency folders and install
COPY pybasicbayes/ /app/pybasicbayes/
COPY pyhsmm/ /app/pyhsmm/
RUN pip install /app/pybasicbayes/ /app/pyhsmm/

# Copy the rest of the code
COPY . .

# Apply legacy patches
RUN python scripts/patch_dependencies.py

# Entry point for the analysis tool
ENTRYPOINT ["python", "-m", "src.chunking"]
CMD ["--help"]
