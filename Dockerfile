# ─────────────────────────────────────────────────────────────
#  FerroMind — Docker image (Node 20 + Python 3.11)
#  Required because Render's Node runtime does not have Python.
# ─────────────────────────────────────────────────────────────

FROM node:20-slim

# Install Python 3 + pip + build tools for native packages (e.g. lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libgomp1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Make 'python' resolve to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ── Install Node deps ─────────────────────────────────────────
COPY package*.json ./
RUN npm ci --omit=dev

# ── Install Python deps ──────────────────────────────────────
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Copy the rest of the project ─────────────────────────────
COPY . .

# Render sets PORT automatically; default to 10000 to match render.yaml
ENV PORT=10000
ENV NODE_ENV=production

EXPOSE 10000

CMD ["node", "server/server.js"]
