# 6G AI Traffic Characterization Testbed
# Multi-stage build for optimized container size

# =============================================================================
# Stage 1: Node.js builder for MCP servers
# =============================================================================
FROM node:20-slim AS node-builder

# Install MCP servers globally
RUN npm install -g \
    @modelcontextprotocol/server-fetch \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-git \
    @modelcontextprotocol/server-memory \
    @modelcontextprotocol/server-puppeteer \
    @modelcontextprotocol/server-brave-search \
    @modelcontextprotocol/server-sequential-thinking \
    && npm cache clean --force

# =============================================================================
# Stage 2: Python application
# =============================================================================
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="6G AI Traffic Testbed"
LABEL description="Testbed for characterizing AI traffic patterns under various network conditions"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NODE_PATH=/usr/local/lib/node_modules \
    PATH="/usr/local/lib/node_modules/.bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Network emulation tools
    iproute2 \
    iptables \
    # Packet capture
    tcpdump \
    libpcap-dev \
    # Network utilities
    curl \
    wget \
    net-tools \
    iputils-ping \
    dnsutils \
    # Git for repository operations
    git \
    # Build tools (for some Python packages)
    build-essential \
    # Node.js runtime (for MCP servers)
    nodejs \
    npm \
    # Chromium for Playwright/Puppeteer
    chromium \
    chromium-driver \
    # Fonts for screenshots
    fonts-liberation \
    fonts-noto-color-emoji \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Node.js global modules from builder
COPY --from=node-builder /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node-builder /usr/local/bin /usr/local/bin

# Set Puppeteer/Playwright to use system Chromium
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=true \
    PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

# Create non-root user for security (but allow sudo for network emulation)
RUN useradd -m -s /bin/bash testbed \
    && apt-get update && apt-get install -y sudo \
    && echo "testbed ALL=(ALL) NOPASSWD: /sbin/tc, /usr/sbin/tcpdump, /sbin/iptables" >> /etc/sudoers \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir playwright \
    && playwright install-deps chromium || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    logs \
    reports/figures \
    capture/captures \
    capture/l7_captures \
    /tmp/testbed-sandbox \
    && chown -R testbed:testbed /app /tmp/testbed-sandbox

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER testbed

# Expose ports
# 8080: mitmproxy for L7 capture
# 8081: mitmproxy web UI
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command: show help
CMD ["--help"]
