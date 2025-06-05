FROM python:3.11-slim

WORKDIR /app

# Update package list and install necessary packages including CA certificates
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    wget \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update CA certificates
RUN update-ca-certificates

# Upgrade pip and install uv for package management
RUN pip install --upgrade pip
RUN pip install uv

COPY . .
# Install the mcp-server-qdrant package
RUN uv pip install --system .

# Expose the default port for SSE transport
EXPOSE 80

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Run the server with SSE transport
CMD ["mcp-server-qdrant", "--transport", "sse"]
